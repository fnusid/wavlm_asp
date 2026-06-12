import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tqdm import tqdm

from model import SpeakerEncoderDualWrapper

import sys

sys.path.append("/home/sidcs/codebase")
from wavlm_single_embedding.model import ECAPA_TDNN


EPS = 1e-8
MIN_WAV_SAMPLES = 512


def l2_normalize_np(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x) + EPS)


def cosine_np(a, b):
    return float(np.dot(l2_normalize_np(a), l2_normalize_np(b)))


def load_audio(path, target_sr=16000):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        raise ValueError(f"{path} has sample rate {sr}, expected {target_sr}")
    return wav


def load_dual_model(ckpt_path, device="cuda", emb_dim=256):
    ckpt = torch.load(ckpt_path, map_location=device)
    filtered = {}
    for key, value in ckpt["state_dict"].items():
        if not key.startswith("model."):
            continue
        key = key.replace("model.", "", 1)
        if key.startswith("single_sp_model.") or key.startswith("arcface_loss."):
            continue
        filtered[key] = value
    model = SpeakerEncoderDualWrapper(emb_dim=emb_dim)
    model.load_state_dict(filtered, strict=True)
    model.to(device).eval()
    return model


def load_teacher_model(ckpt_path, device="cuda"):
    model = ECAPA_TDNN(C=1024).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    filtered = {}
    for key, value in ckpt["state_dict"].items():
        if not key.startswith("model."):
            continue
        if "arcface" in key or "arc_face" in key:
            continue
        filtered[key.replace("model.", "", 1)] = value
    model.load_state_dict(filtered, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def get_teacher_embedding(model, wav, device):
    if wav.shape[0] < MIN_WAV_SAMPLES:
        wav = np.pad(wav, (0, MIN_WAV_SAMPLES - wav.shape[0]))
    wav = torch.from_numpy(wav.astype(np.float32)).to(device).unsqueeze(0)
    emb = model(wav).squeeze(0)
    emb = F.normalize(emb, dim=0)
    return emb.cpu().numpy()


@torch.no_grad()
def get_dual_embeddings(model, wav, device):
    if wav.shape[0] < MIN_WAV_SAMPLES:
        wav = np.pad(wav, (0, MIN_WAV_SAMPLES - wav.shape[0]))
    wav = torch.from_numpy(wav.astype(np.float32)).to(device).unsqueeze(0)
    emb = model(wav).squeeze(0)
    emb = F.normalize(emb, dim=-1)
    return emb.cpu().numpy()


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
    def __init__(self, threshold=0.55, ema_alpha=None, pending_threshold=0.60, pending_confirm_hits=2, pending_max_age=5):
        self.threshold = threshold
        self.ema_alpha = ema_alpha
        self.pending_threshold = pending_threshold
        self.pending_confirm_hits = pending_confirm_hits
        self.pending_max_age = pending_max_age
        self.centroids = []
        self.counts = []
        self.pending = []
        self.next_pending_id = 0

    def best_match(self, emb):
        if len(self.centroids) == 0:
            return None, -1.0
        sims = [cosine_np(emb, c) for c in self.centroids]
        idx = int(np.argmax(sims))
        return idx, float(sims[idx])

    def best_pending_match(self, emb):
        if len(self.pending) == 0:
            return None, -1.0
        sims = [cosine_np(emb, p["centroid"]) for p in self.pending]
        idx = int(np.argmax(sims))
        return idx, float(sims[idx])

    def cleanup_pending(self, global_step):
        self.pending = [p for p in self.pending if (global_step - p["last_seen"]) <= self.pending_max_age]

    def update_confirmed(self, mem_id, emb):
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
        self.centroids.append(l2_normalize_np(emb))
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
        pending_id = p["pending_id"]
        pending_count = p["count"]
        self.pending.pop(pending_idx)
        return mem_id, pending_id, pending_count

    def assign(self, emb, global_step, blocked_ids=None):
        blocked_ids = blocked_ids or set()
        emb = l2_normalize_np(emb)
        self.cleanup_pending(global_step)

        best_id = None
        best_sim = -1.0
        for idx, c in enumerate(self.centroids):
            if idx in blocked_ids:
                continue
            sim = cosine_np(emb, c)
            if sim > best_sim:
                best_sim = sim
                best_id = idx

        if best_id is not None and best_sim >= self.threshold:
            self.update_confirmed(best_id, emb)
            return {
                "mem_id": best_id,
                "best_sim": best_sim,
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
                    "status": "promoted_pending",
                    "pending_id": pending_id,
                    "pending_count": pending_count,
                }
            return {
                "mem_id": -1,
                "best_sim": best_sim,
                "status": "matched_pending",
                "pending_id": p["pending_id"],
                "pending_count": p["count"],
            }

        pending_id = self.add_pending(emb, global_step)
        return {
            "mem_id": -1,
            "best_sim": best_sim,
            "status": "new_pending",
            "pending_id": pending_id,
            "pending_count": 1,
        }


def assign_two_embeddings(memory, emb_a, emb_b, global_step):
    used = set()
    out_a = memory.assign(emb_a, global_step, blocked_ids=used)
    if out_a["mem_id"] >= 0:
        used.add(out_a["mem_id"])
    out_b = memory.assign(emb_b, global_step + 1, blocked_ids=used)
    return out_a, out_b


def score_views(assignments_df):
    confirmed = assignments_df[assignments_df["pred_memory_id"].astype(int) >= 0].copy()
    all_out = assignments_df.copy()
    all_out["pred_label"] = all_out.apply(
        lambda r: f"mem_{int(r['pred_memory_id'])}"
        if int(r["pred_memory_id"]) >= 0
        else (f"pending_{int(r['pending_id'])}" if pd.notna(r["pending_id"]) else None),
        axis=1,
    )
    all_out = all_out[all_out["pred_label"].notna()].copy()

    def compute(df, pred_col):
        if len(df) == 0:
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
        true_labels = df["true_speaker"].astype(str).values
        pred_labels = df[pred_col].astype(str).values
        segment_ids = df["segment_id"].astype(int).values
        switches, transitions, switch_rate = compute_id_switches(true_labels, pred_labels, segment_ids)
        return {
            "num_eval_records": int(len(df)),
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

    return compute(confirmed, "pred_memory_id"), compute(all_out, "pred_label"), confirmed, all_out


def build_speaker_anchors(speakers_csv, mix, teacher_model, args):
    speakers_df = pd.read_csv(speakers_csv)
    anchors = {}
    for _, row in speakers_df.iterrows():
        spk = str(row["speaker_id"])
        start = int(round(float(row["anchor_start_sec"]) * args.sample_rate))
        end = int(round(float(row["anchor_end_sec"]) * args.sample_rate))
        chunk = mix[start:end]
        if chunk.size == 0:
            continue
        anchors[spk] = get_teacher_embedding(teacher_model, chunk, args.device)
    return anchors


def build_interval_events(assignments_df, include_pending):
    events = []
    for _, row in assignments_df.iterrows():
        if int(row["pred_memory_id"]) >= 0:
            pred_label = f"mem_{int(row['pred_memory_id'])}"
        elif include_pending and pd.notna(row["pending_id"]):
            pred_label = f"pending_{int(row['pending_id'])}"
        else:
            pred_label = None
        events.append(
            {
                "start": float(row["start_sec"]),
                "end": float(row["end_sec"]),
                "ref": str(row["true_speaker"]),
                "hyp": pred_label,
            }
        )
    return events


def compute_der_from_assignments(assignments_df, include_pending):
    events = build_interval_events(assignments_df, include_pending=include_pending)
    if not events:
        return {"der": 0.0, "ref_time": 0.0, "miss": 0.0, "fa": 0.0, "conf": 0.0}

    boundaries = sorted(set([e["start"] for e in events] + [e["end"] for e in events]))
    ref_labels = sorted({e["ref"] for e in events})
    hyp_labels = sorted({e["hyp"] for e in events if e["hyp"] is not None})
    overlap = np.zeros((len(hyp_labels), len(ref_labels)), dtype=np.float64)
    hyp_index = {h: i for i, h in enumerate(hyp_labels)}
    ref_index = {r: i for i, r in enumerate(ref_labels)}

    for i in range(len(boundaries) - 1):
        t0, t1 = boundaries[i], boundaries[i + 1]
        dt = t1 - t0
        if dt <= 0:
            continue
        ref_set = {e["ref"] for e in events if e["start"] <= t0 + 1e-9 and e["end"] >= t1 - 1e-9}
        hyp_set = {e["hyp"] for e in events if e["hyp"] is not None and e["start"] <= t0 + 1e-9 and e["end"] >= t1 - 1e-9}
        for h in hyp_set:
            for r in ref_set:
                overlap[hyp_index[h], ref_index[r]] += dt

    mapping = {}
    if overlap.size > 0:
        rows, cols = linear_sum_assignment(-overlap)
        mapping = {hyp_labels[r]: ref_labels[c] for r, c in zip(rows, cols)}

    miss = 0.0
    fa = 0.0
    conf = 0.0
    ref_time = 0.0
    for i in range(len(boundaries) - 1):
        t0, t1 = boundaries[i], boundaries[i + 1]
        dt = t1 - t0
        if dt <= 0:
            continue
        ref_set = {e["ref"] for e in events if e["start"] <= t0 + 1e-9 and e["end"] >= t1 - 1e-9}
        hyp_set = {mapping[h] for e in events for h in ([e["hyp"]] if e["hyp"] is not None else []) if e["start"] <= t0 + 1e-9 and e["end"] >= t1 - 1e-9 and h in mapping}
        ref_n = len(ref_set)
        hyp_n = len(hyp_set)
        ref_time += dt * ref_n
        correct = len(ref_set & hyp_set)
        miss += dt * max(0, ref_n - hyp_n)
        fa += dt * max(0, hyp_n - ref_n)
        conf += dt * (min(ref_n, hyp_n) - correct)

    der = (miss + fa + conf) / max(ref_time, EPS)
    return {"der": der, "ref_time": ref_time, "miss": miss, "fa": fa, "conf": conf}


def evaluate_one_meeting(meeting_row, dual_model, teacher_model, args):
    mix = load_audio(meeting_row["mix_path"], target_sr=args.sample_rate)
    timeline_df = pd.read_csv(meeting_row["timeline_csv"])
    speaker_anchors = build_speaker_anchors(meeting_row["speakers_csv"], mix, teacher_model, args)
    memory = SpeakerMemoryBank(
        threshold=args.threshold,
        ema_alpha=args.ema_alpha,
        pending_threshold=args.pending_threshold,
        pending_confirm_hits=args.pending_confirm_hits,
        pending_max_age=args.pending_max_age,
    )

    records = []
    global_step = 0
    for _, row in timeline_df.iterrows():
        start_sec = float(row["start_sec"])
        end_sec = float(row["end_sec"])
        start = int(round(start_sec * args.sample_rate))
        end = int(round(end_sec * args.sample_rate))
        chunk = mix[start:end]
        if chunk.size == 0:
            continue

        pred = get_dual_embeddings(dual_model, chunk, args.device)
        num_active = int(row["num_active"])

        if num_active == 1:
            best0 = memory.best_match(pred[0])[1] if len(memory.centroids) > 0 else -1.0
            best1 = memory.best_match(pred[1])[1] if len(memory.centroids) > 0 else -1.0
            selected_idx = 0 if best0 >= best1 else 1
            out = memory.assign(pred[selected_idx], global_step)
            records.append(
                {
                    "meeting_id": meeting_row["meeting_id"],
                    "segment_id": int(row["segment_id"]),
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "true_speaker": str(row["active_speakers"]),
                    "pred_memory_id": int(out["mem_id"]),
                    "num_active": 1,
                    "selected_embedding_idx": selected_idx,
                    "ignored_embedding_idx": 1 - selected_idx,
                    "memory_match_score": float(out["best_sim"]),
                    "assignment_status": out["status"],
                    "pending_id": out["pending_id"],
                    "pending_count": out["pending_count"],
                }
            )
            global_step += 1
            continue

        spk1 = str(row["speaker_1_id"])
        spk2 = str(row["speaker_2_id"])
        t1 = speaker_anchors.get(spk1)
        t2 = speaker_anchors.get(spk2)
        if t1 is not None and t2 is not None:
            direct = cosine_np(pred[0], t1) + cosine_np(pred[1], t2)
            swap = cosine_np(pred[0], t2) + cosine_np(pred[1], t1)
        else:
            direct = 0.0
            swap = -1.0
        if direct >= swap:
            labeled = [(pred[0], spk1, 0), (pred[1], spk2, 1)]
        else:
            labeled = [(pred[0], spk2, 0), (pred[1], spk1, 1)]

        out_a, out_b = assign_two_embeddings(memory, labeled[0][0], labeled[1][0], global_step)
        pair_outs = [out_a, out_b]
        for (emb, true_spk, emb_idx), out in zip(labeled, pair_outs):
            records.append(
                {
                    "meeting_id": meeting_row["meeting_id"],
                    "segment_id": int(row["segment_id"]),
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "true_speaker": true_spk,
                    "pred_memory_id": int(out["mem_id"]),
                    "num_active": 2,
                    "selected_embedding_idx": emb_idx,
                    "ignored_embedding_idx": "",
                    "memory_match_score": float(out["best_sim"]),
                    "assignment_status": out["status"],
                    "pending_id": out["pending_id"],
                    "pending_count": out["pending_count"],
                }
            )
        global_step += 2

    assignments_df = pd.DataFrame(records)
    conf_metrics, all_metrics, _, _ = score_views(assignments_df)
    der_confirmed = compute_der_from_assignments(assignments_df, include_pending=False)
    der_all = compute_der_from_assignments(assignments_df, include_pending=True)
    summary = {
        "meeting_id": meeting_row["meeting_id"],
        "num_records_total": int(len(assignments_df)),
        "num_confirmed_records": int((assignments_df["pred_memory_id"].astype(int) >= 0).sum()),
        "confirmation_rate": float((assignments_df["pred_memory_id"].astype(int) >= 0).sum() / max(1, len(assignments_df))),
        "memory_bank_size": int(len(memory.centroids)),
        "pending_left": int(len(memory.pending)),
    }
    der_row = {
        "meeting_id": meeting_row["meeting_id"],
        "confirmed_only_der": der_confirmed["der"],
        "confirmed_only_ref_time": der_confirmed["ref_time"],
        "confirmed_only_miss": der_confirmed["miss"],
        "confirmed_only_fa": der_confirmed["fa"],
        "confirmed_only_conf": der_confirmed["conf"],
        "all_output_der": der_all["der"],
        "all_output_ref_time": der_all["ref_time"],
        "all_output_miss": der_all["miss"],
        "all_output_fa": der_all["fa"],
        "all_output_conf": der_all["conf"],
    }
    picovoice_row = {
        "meeting_id": meeting_row["meeting_id"],
        "der": der_all["der"],
        "ref_time": der_all["ref_time"],
        "miss": der_all["miss"],
        "other": der_all["fa"] + der_all["conf"],
    }
    return assignments_df, {**summary, **conf_metrics}, {**summary, **all_metrics}, der_row, picovoice_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_csv", type=str, default="/home/sidcs/datasets/LibriMix/long_meetings_test/manifest.csv")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--dual_ckpt", type=str, required=True)
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--meeting_id", type=str, default=None)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
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

    dual_model = load_dual_model(args.dual_ckpt, device=args.device)
    teacher_model = load_teacher_model(args.teacher_ckpt, device=args.device)

    assignments_rows = []
    confirmed_rows = []
    all_output_rows = []
    der_rows = []
    picovoice_rows = []

    for _, meeting_row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Evaluating meetings"):
        assignments_df, conf_row, all_row, der_row, picovoice_row = evaluate_one_meeting(meeting_row, dual_model, teacher_model, args)
        assignments_rows.extend(assignments_df.to_dict("records"))
        confirmed_rows.append(conf_row)
        all_output_rows.append(all_row)
        der_rows.append(der_row)
        picovoice_rows.append(picovoice_row)

    assignments_out = pd.DataFrame(assignments_rows)
    confirmed_out = pd.DataFrame(confirmed_rows)
    all_output_out = pd.DataFrame(all_output_rows)
    der_out = pd.DataFrame(der_rows)
    picovoice_out = pd.DataFrame(picovoice_rows)

    assignments_out.to_csv(out_dir / "meeting_memory_assignments.csv", index=False)
    confirmed_out.to_csv(out_dir / "meeting_memory_confirmed_metrics.csv", index=False)
    all_output_out.to_csv(out_dir / "meeting_memory_all_output_metrics.csv", index=False)
    confirmed_out.to_csv(out_dir / "meeting_memory_metrics.csv", index=False)
    all_output_out.to_csv(out_dir / "meeting_memory_metrics_all_output.csv", index=False)
    der_out.to_csv(out_dir / "meeting_memory_der.csv", index=False)
    picovoice_out.to_csv(out_dir / "meeting_memory_der_picovoice_style.csv", index=False)

    print("\n=== Confirmed mean ===")
    print(confirmed_out.mean(numeric_only=True).to_string())
    print("\n=== All-output mean ===")
    print(all_output_out.mean(numeric_only=True).to_string())
    print("\n=== DER mean ===")
    print(der_out.mean(numeric_only=True).to_string())


if __name__ == "__main__":
    main()
