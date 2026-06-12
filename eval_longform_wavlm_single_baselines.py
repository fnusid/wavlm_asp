import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tqdm import tqdm

import sys

sys.path.append("/home/sidcs/codebase")
from wavlm_single_embedding.model import SpeakerEncoderWrapper


EPS = 1e-8


def l2_normalize_np(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x) + EPS)


def cosine_np(a, b):
    a = l2_normalize_np(a)
    b = l2_normalize_np(b)
    return float(np.dot(a, b))


def load_audio(path, target_sr=16000):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        raise ValueError(f"{path} has sample rate {sr}, expected {target_sr}")
    return wav


def load_single_model(ckpt_path, emb_dim=256, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device)
    filtered = {}
    for key, value in ckpt["state_dict"].items():
        if not key.startswith("model."):
            continue
        if "arcface" in key or "arc_face" in key:
            continue
        filtered[key.replace("model.", "", 1)] = value
    model = SpeakerEncoderWrapper(emb_dim=emb_dim)
    model.load_state_dict(filtered, strict=True)
    model.to(device).eval()
    return model


def cluster_accuracy(pred_labels, true_labels):
    pred = np.asarray(pred_labels)
    true = np.asarray(true_labels)
    total = 0
    for cluster_id in np.unique(pred):
        mask = pred == cluster_id
        subset = true[mask]
        if len(subset) == 0:
            continue
        total += Counter(subset).most_common(1)[0][1]
    return total / max(1, len(true))


def compute_id_switches(true_labels, pred_ids, segment_ids):
    by_spk = defaultdict(list)
    for true_spk, pred_id, seg_id in zip(true_labels, pred_ids, segment_ids):
        by_spk[true_spk].append((seg_id, pred_id))

    switches = 0
    transitions = 0
    for _, items in by_spk.items():
        items = sorted(items, key=lambda x: x[0])
        prev = None
        for _, pred_id in items:
            if prev is not None:
                transitions += 1
                if pred_id != prev:
                    switches += 1
            prev = pred_id

    return switches, transitions, switches / max(1, transitions)


def compute_reid_accuracy(true_labels, pred_ids):
    by_spk = defaultdict(list)
    for true_spk, pred_id in zip(true_labels, pred_ids):
        by_spk[true_spk].append(pred_id)

    correct = 0
    total = 0
    for _, assigned_ids in by_spk.items():
        if len(assigned_ids) < 2:
            continue
        majority_id = Counter(assigned_ids).most_common(1)[0][0]
        for pred_id in assigned_ids:
            total += 1
            if pred_id == majority_id:
                correct += 1

    return correct / max(1, total)


class SpeakerMemoryBank:
    def __init__(
        self,
        threshold=0.55,
        ema_alpha=None,
        pending_threshold=0.60,
        pending_confirm_hits=2,
        pending_max_age=5,
    ):
        self.threshold = threshold
        self.ema_alpha = ema_alpha
        self.pending_threshold = pending_threshold
        self.pending_confirm_hits = pending_confirm_hits
        self.pending_max_age = pending_max_age

        self.centroids = []
        self.counts = []
        self.pending = []
        self.next_pending_id = 0

    def __len__(self):
        return len(self.centroids)

    def best_match(self, emb):
        emb = l2_normalize_np(emb)
        if len(self.centroids) == 0:
            return None, -1.0
        sims = [cosine_np(emb, c) for c in self.centroids]
        idx = int(np.argmax(sims))
        return idx, float(sims[idx])

    def best_pending_match(self, emb):
        emb = l2_normalize_np(emb)
        if len(self.pending) == 0:
            return None, -1.0
        sims = [cosine_np(emb, p["centroid"]) for p in self.pending]
        idx = int(np.argmax(sims))
        return idx, float(sims[idx])

    def cleanup_pending(self, global_step):
        self.pending = [
            p for p in self.pending if (global_step - p["last_seen"]) <= self.pending_max_age
        ]

    def update_confirmed(self, mem_id, emb):
        emb = l2_normalize_np(emb)
        old = self.centroids[mem_id]
        if self.ema_alpha is None:
            n = self.counts[mem_id]
            new = (old * n + emb) / (n + 1)
        else:
            a = self.ema_alpha
            new = a * old + (1.0 - a) * emb
        self.centroids[mem_id] = l2_normalize_np(new)
        self.counts[mem_id] += 1

    def add_confirmed(self, emb):
        emb = l2_normalize_np(emb)
        self.centroids.append(emb)
        self.counts.append(1)
        return len(self.centroids) - 1

    def add_pending(self, emb, global_step):
        pending_id = self.next_pending_id
        self.next_pending_id += 1
        self.pending.append(
            {
                "pending_id": pending_id,
                "centroid": l2_normalize_np(emb),
                "count": 1,
                "last_seen": global_step,
            }
        )
        return pending_id

    def update_pending(self, pending_idx, emb, global_step):
        p = self.pending[pending_idx]
        n = p["count"]
        new = (p["centroid"] * n + l2_normalize_np(emb)) / (n + 1)
        p["centroid"] = l2_normalize_np(new)
        p["count"] += 1
        p["last_seen"] = global_step
        return p

    def promote_pending(self, pending_idx):
        p = self.pending[pending_idx]
        mem_id = self.add_confirmed(p["centroid"])
        promoted_pending_id = p["pending_id"]
        promoted_count = p["count"]
        self.pending.pop(pending_idx)
        return mem_id, promoted_pending_id, promoted_count

    def assign(self, emb, global_step):
        emb = l2_normalize_np(emb)
        self.cleanup_pending(global_step)

        best_id, best_sim = self.best_match(emb)
        if best_id is not None and best_sim >= self.threshold:
            self.update_confirmed(best_id, emb)
            return {
                "mem_id": best_id,
                "best_sim": best_sim,
                "is_new": False,
                "status": "matched_confirmed",
                "pending_id": None,
                "pending_count": None,
            }

        pending_idx, pending_sim = self.best_pending_match(emb)
        if pending_idx is not None and pending_sim >= self.pending_threshold:
            p = self.update_pending(pending_idx, emb, global_step)
            if p["count"] >= self.pending_confirm_hits:
                mem_id, pending_id, pending_count = self.promote_pending(pending_idx)
                return {
                    "mem_id": mem_id,
                    "best_sim": best_sim,
                    "is_new": True,
                    "status": "promoted_pending",
                    "pending_id": pending_id,
                    "pending_count": pending_count,
                }
            return {
                "mem_id": -1,
                "best_sim": best_sim,
                "is_new": False,
                "status": "matched_pending",
                "pending_id": p["pending_id"],
                "pending_count": p["count"],
            }

        pending_id = self.add_pending(emb, global_step)
        return {
            "mem_id": -1,
            "best_sim": best_sim,
            "is_new": False,
            "status": "new_pending",
            "pending_id": pending_id,
            "pending_count": 1,
        }


def embed_chunks(model, chunks, batch_size, device):
    all_embs = []
    for start in range(0, len(chunks), batch_size):
        batch = torch.stack(
            [torch.from_numpy(chunk.astype(np.float32)) for chunk in chunks[start : start + batch_size]],
            dim=0,
        ).to(device)
        with torch.no_grad():
            emb = model(batch)
            emb = F.normalize(emb, dim=-1)
        all_embs.append(emb.cpu().numpy())
    return np.concatenate(all_embs, axis=0)


def extract_oracle_source_segments(meeting_row, sample_rate):
    speakers_df = pd.read_csv(meeting_row["speakers_csv"])
    speaker_tracks = {}
    for _, row in speakers_df.iterrows():
        speaker_tracks[str(row["speaker_id"])] = load_audio(row["speaker_ref_path"], target_sr=sample_rate)

    timeline_df = pd.read_csv(meeting_row["timeline_csv"])
    records = []
    chunks = []
    for _, row in timeline_df.iterrows():
        start_sec = float(row["start_sec"])
        end_sec = float(row["end_sec"])
        start = int(round(start_sec * sample_rate))
        end = int(round(end_sec * sample_rate))
        active_speakers = str(row["active_speakers"]).split("|")
        for true_speaker in active_speakers:
            track = speaker_tracks[str(true_speaker)]
            chunk = track[start:end]
            if chunk.size == 0:
                continue
            records.append(
                {
                    "meeting_id": meeting_row["meeting_id"],
                    "segment_id": int(row["segment_id"]),
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "num_active": int(row["num_active"]),
                    "true_speaker": str(true_speaker),
                }
            )
            chunks.append(chunk)
    return records, chunks


def evaluate_kmeans(meeting_row, records, embs, oracle_num_speakers):
    true_labels = np.asarray([r["true_speaker"] for r in records])
    n_clusters = int(meeting_row["num_global_speakers"]) if oracle_num_speakers else len(np.unique(true_labels))
    n_clusters = min(n_clusters, len(records))
    pred_ids = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(embs)
    segment_ids = np.asarray([r["segment_id"] for r in records])
    switches, transitions, switch_rate = compute_id_switches(true_labels, pred_ids, segment_ids)
    metrics = {
        "meeting_id": meeting_row["meeting_id"],
        "num_records": len(records),
        "num_true_speakers_eval": int(len(np.unique(true_labels))),
        "num_global_speakers_manifest": int(meeting_row["num_global_speakers"]),
        "num_pred_clusters": int(len(np.unique(pred_ids))),
        "nmi": float(normalized_mutual_info_score(true_labels, pred_ids)),
        "ari": float(adjusted_rand_score(true_labels, pred_ids)),
        "cluster_acc": float(cluster_accuracy(pred_ids, true_labels)),
        "id_switches": int(switches),
        "id_transitions": int(transitions),
        "id_switch_rate": float(switch_rate),
        "reid_accuracy": float(compute_reid_accuracy(true_labels, pred_ids)),
    }
    rows = []
    for rec, pred_id in zip(records, pred_ids):
        row = dict(rec)
        row["pred_cluster_id"] = int(pred_id)
        rows.append(row)
    return metrics, rows


def confirmed_df(assignments_df):
    return assignments_df[assignments_df["pred_memory_id"].astype(int) >= 0].copy()


def all_output_df(assignments_df):
    df = assignments_df.copy()
    labels = []
    for _, row in df.iterrows():
        if int(row["pred_memory_id"]) >= 0:
            labels.append(f"mem_{int(row['pred_memory_id'])}")
        elif pd.notna(row["pending_id"]):
            labels.append(f"pending_{int(row['pending_id'])}")
        else:
            labels.append(None)
    df["pred_label"] = labels
    return df[df["pred_label"].notna()].copy()


def metrics_from_assignment_view(base_df, pred_col):
    if len(base_df) == 0:
        return {
            "num_eval_records": 0,
            "num_true_speakers": 0,
            "num_output_speakers": 0,
            "nmi": 0.0,
            "ari": 0.0,
            "cluster_acc": 0.0,
            "id_switches": 0,
            "id_transitions": 0,
            "id_switch_rate": 0.0,
            "reid_accuracy": 0.0,
        }
    true_labels = base_df["true_speaker"].astype(str).values
    pred_labels = base_df[pred_col].astype(str).values
    segment_ids = base_df["segment_id"].astype(int).values
    switches, transitions, switch_rate = compute_id_switches(true_labels, pred_labels, segment_ids)
    return {
        "num_eval_records": int(len(base_df)),
        "num_true_speakers": int(len(np.unique(true_labels))),
        "num_output_speakers": int(len(np.unique(pred_labels))),
        "nmi": float(normalized_mutual_info_score(true_labels, pred_labels)),
        "ari": float(adjusted_rand_score(true_labels, pred_labels)),
        "cluster_acc": float(cluster_accuracy(pred_labels, true_labels)),
        "id_switches": int(switches),
        "id_transitions": int(transitions),
        "id_switch_rate": float(switch_rate),
        "reid_accuracy": float(compute_reid_accuracy(true_labels, pred_labels)),
    }


def evaluate_memory(records, embs, args):
    memory = SpeakerMemoryBank(
        threshold=args.threshold,
        ema_alpha=args.ema_alpha,
        pending_threshold=args.pending_threshold,
        pending_confirm_hits=args.pending_confirm_hits,
        pending_max_age=args.pending_max_age,
    )
    rows = []
    for global_step, (rec, emb) in enumerate(zip(records, embs)):
        out = memory.assign(emb, global_step)
        row = dict(rec)
        row["pred_memory_id"] = int(out["mem_id"])
        row["memory_match_score"] = float(out["best_sim"])
        row["created_new_memory"] = bool(out["is_new"])
        row["assignment_status"] = out["status"]
        row["pending_id"] = out["pending_id"]
        row["pending_count"] = out["pending_count"]
        rows.append(row)

    assignments_df = pd.DataFrame(rows)
    conf = confirmed_df(assignments_df)
    all_out = all_output_df(assignments_df)
    conf_metrics = metrics_from_assignment_view(conf, "pred_memory_id")
    all_metrics = metrics_from_assignment_view(all_out, "pred_label")
    summary = {
        "num_records_total": int(len(assignments_df)),
        "num_confirmed_records": int(len(conf)),
        "confirmation_rate": float(len(conf) / max(1, len(assignments_df))),
        "memory_bank_size": int(len(memory.centroids)),
        "pending_left": int(len(memory.pending)),
    }
    return assignments_df, conf_metrics, all_metrics, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest_csv",
        type=str,
        default="/home/sidcs/datasets/LibriMix/long_meetings_test/manifest.csv",
    )
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--single_ckpt", type=str, required=True)
    parser.add_argument("--meeting_id", type=str, default=None)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--oracle_num_speakers", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--ema_alpha", type=float, default=None)
    parser.add_argument("--pending_threshold", type=float, default=0.60)
    parser.add_argument("--pending_confirm_hits", type=int, default=2)
    parser.add_argument("--pending_max_age", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_csv(args.manifest_csv)
    if args.meeting_id is not None:
        manifest_df = manifest_df[manifest_df["meeting_id"].astype(str) == str(args.meeting_id)].copy()

    model = load_single_model(args.single_ckpt, device=args.device)

    kmeans_metrics_rows = []
    kmeans_assignment_rows = []
    memory_confirmed_rows = []
    memory_all_rows = []
    memory_assignment_rows = []

    for _, meeting_row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Evaluating meetings"):
        records, chunks = extract_oracle_source_segments(meeting_row, args.sample_rate)
        if not records:
            continue
        embs = embed_chunks(model, chunks, args.batch_size, args.device)
        embs = np.asarray([l2_normalize_np(x) for x in embs], dtype=np.float32)

        kmeans_metrics, kmeans_rows = evaluate_kmeans(meeting_row, records, embs, args.oracle_num_speakers)
        kmeans_metrics_rows.append(kmeans_metrics)
        kmeans_assignment_rows.extend(kmeans_rows)

        mem_assign_df, mem_conf_metrics, mem_all_metrics, mem_summary = evaluate_memory(records, embs, args)
        mem_assign_df["meeting_id"] = meeting_row["meeting_id"]
        memory_assignment_rows.extend(mem_assign_df.to_dict("records"))

        conf_row = {"meeting_id": meeting_row["meeting_id"], **mem_summary, **mem_conf_metrics}
        all_row = {"meeting_id": meeting_row["meeting_id"], **mem_summary, **mem_all_metrics}
        memory_confirmed_rows.append(conf_row)
        memory_all_rows.append(all_row)

    kmeans_metrics_df = pd.DataFrame(kmeans_metrics_rows)
    kmeans_assignments_df = pd.DataFrame(kmeans_assignment_rows)
    memory_confirmed_df_out = pd.DataFrame(memory_confirmed_rows)
    memory_all_df_out = pd.DataFrame(memory_all_rows)
    memory_assignments_df = pd.DataFrame(memory_assignment_rows)

    prefix = "wavlm_frozen_single"
    kmeans_metrics_df.to_csv(out_dir / f"{prefix}_kmeans_metrics.csv", index=False)
    kmeans_assignments_df.to_csv(out_dir / f"{prefix}_kmeans_assignments.csv", index=False)
    memory_confirmed_df_out.to_csv(out_dir / f"{prefix}_memory_confirmed_metrics.csv", index=False)
    memory_all_df_out.to_csv(out_dir / f"{prefix}_memory_all_output_metrics.csv", index=False)
    memory_assignments_df.to_csv(out_dir / f"{prefix}_memory_assignments.csv", index=False)

    if len(kmeans_metrics_df) > 0:
        print("\n=== KMeans mean ===")
        print(kmeans_metrics_df.mean(numeric_only=True).to_string())
    if len(memory_confirmed_df_out) > 0:
        print("\n=== Memory confirmed mean ===")
        print(memory_confirmed_df_out.mean(numeric_only=True).to_string())
    if len(memory_all_df_out) > 0:
        print("\n=== Memory all-output mean ===")
        print(memory_all_df_out.mean(numeric_only=True).to_string())


if __name__ == "__main__":
    main()
