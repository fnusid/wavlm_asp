import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


# ------------------------------------------------------------
# Your model imports
# ------------------------------------------------------------
sys.path.append("/home/sidcs/codebase/wavlm_dual_embedding/")
from model import SpeakerEncoderDualWrapper

sys.path.append("/home/sidcs/codebase/")
from wavlm_single_embedding.model import SpeakerEncoderWrapper as SingleSpkEncoder


EPS = 1e-8


# ============================================================
# Model loading helpers
# ============================================================
def joint_trained_model_weights(state):
    new_state = {}
    for k, v in state.items():
        if k.startswith("dual_emb_model."):
            k2 = k.replace("dual_emb_model.", "")
            new_state[k2] = v
    return new_state


def strip_dual_model_weights(state):
    new_state = {}
    for k, v in state.items():
        if not k.startswith("model."):
            continue
        k2 = k.replace("model.", "")
        if k2.startswith("single_sp_model.") or k2.startswith("arcface_loss."):
            continue
        new_state[k2] = v
    return new_state


def load_dual_model(ckpt_path, joint_ckpt_path=None, emb_dim=256, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = strip_dual_model_weights(ckpt["state_dict"])

    model = SpeakerEncoderDualWrapper(emb_dim=emb_dim)
    model.load_state_dict(state, strict=True)

    if joint_ckpt_path is not None:
        joint_ckpt = torch.load(joint_ckpt_path, map_location=device)
        joint_state = joint_trained_model_weights(joint_ckpt["state_dict"])
        model.load_state_dict(joint_state, strict=True)

    model.to(device).eval()
    return model


def load_teacher_model(teacher_ckpt, device="cuda"):
    teacher = SingleSpkEncoder().to(device)

    ckpt = torch.load(teacher_ckpt, map_location=device)
    state = ckpt["state_dict"]

    filtered = {}
    for k, v in state.items():
        if not k.startswith("model."):
            continue
        if "arcface" in k or "arc_face" in k:
            continue
        new_k = k.replace("model.", "", 1)
        filtered[new_k] = v

    teacher.load_state_dict(filtered, strict=True)
    teacher.eval()

    for p in teacher.parameters():
        p.requires_grad = False

    return teacher


# ============================================================
# Basic embedding helpers
# ============================================================
def l2_normalize_np(x):
    x = np.asarray(x)
    return x / (np.linalg.norm(x) + EPS)


def cosine_np(a, b):
    a = l2_normalize_np(a)
    b = l2_normalize_np(b)
    return float(np.dot(a, b))


@torch.no_grad()
def get_teacher_emb_from_ref_track(
    teacher_model,
    wav_path,
    start_sec,
    end_sec,
    device="cuda",
    sr=16000,
):
    wav, wav_sr = torchaudio.load(wav_path)
    wav = wav.mean(0)

    if wav_sr != sr:
        wav = torchaudio.functional.resample(wav, wav_sr, sr)

    s = int(start_sec * sr)
    e = int(end_sec * sr)
    chunk = wav[s:e]

    if chunk.numel() < int(0.5 * sr):
        return None

    if chunk.abs().mean().item() < 1e-5:
        return None

    chunk = chunk.to(device).unsqueeze(0)
    emb = teacher_model(chunk).squeeze(0)
    emb = F.normalize(emb, dim=0)

    return emb


@torch.no_grad()
def get_dual_embeddings(dual_model, wav, device="cuda"):
    """
    wav: [T]
    returns numpy [2, D]
    """
    wav = wav.to(device).unsqueeze(0)

    emb = dual_model(wav)  # [1, 2, D]
    emb = emb.squeeze(0)
    emb = F.normalize(emb, dim=-1)

    return emb.detach().cpu().numpy()


# ============================================================
# Memory bank with pending-speaker confirmation
# ============================================================
class SpeakerMemoryBank:
    def __init__(
        self,
        threshold=0.55,
        ema_alpha=None,
        pending_threshold=0.60,
        pending_confirm_hits=2,
        pending_max_age=5,
    ):
        """
        threshold:
            cosine threshold for matching to existing confirmed memory.

        ema_alpha:
            if None, use count-based running average.
            if float, use EMA:
                centroid = alpha * centroid + (1-alpha) * new_embedding

        pending_threshold:
            cosine threshold for matching an unmatched embedding to
            an existing pending candidate.

        pending_confirm_hits:
            number of observations required before a pending candidate
            becomes a confirmed memory speaker.

        pending_max_age:
            pending candidates older than this many global steps are discarded.
        """
        self.threshold = threshold
        self.ema_alpha = ema_alpha

        self.pending_threshold = pending_threshold
        self.pending_confirm_hits = pending_confirm_hits
        self.pending_max_age = pending_max_age

        self.centroids = []
        self.counts = []
        self.created_at = []

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
        kept = []
        for p in self.pending:
            age = global_step - p["last_seen"]
            if age <= self.pending_max_age:
                kept.append(p)
        self.pending = kept

    def update_confirmed(self, mem_id, emb):
        emb = l2_normalize_np(emb)
        old = self.centroids[mem_id]

        if self.ema_alpha is None:
            n = self.counts[mem_id]
            new = (old * n + emb) / (n + 1)
            self.counts[mem_id] += 1
        else:
            a = self.ema_alpha
            new = a * old + (1.0 - a) * emb
            self.counts[mem_id] += 1

        self.centroids[mem_id] = l2_normalize_np(new)

    def add_confirmed(self, emb, global_step):
        emb = l2_normalize_np(emb)
        self.centroids.append(emb)
        self.counts.append(1)
        self.created_at.append(global_step)
        return len(self.centroids) - 1

    def add_pending(self, emb, global_step):
        emb = l2_normalize_np(emb)

        pending_id = self.next_pending_id
        self.next_pending_id += 1

        self.pending.append({
            "pending_id": pending_id,
            "centroid": emb,
            "count": 1,
            "first_seen": global_step,
            "last_seen": global_step,
        })

        return pending_id

    def update_pending(self, pending_idx, emb, global_step):
        emb = l2_normalize_np(emb)
        p = self.pending[pending_idx]

        n = p["count"]
        new = (p["centroid"] * n + emb) / (n + 1)

        p["centroid"] = l2_normalize_np(new)
        p["count"] += 1
        p["last_seen"] = global_step

        return p

    def promote_pending(self, pending_idx, global_step):
        p = self.pending[pending_idx]
        mem_id = self.add_confirmed(p["centroid"], global_step)

        promoted_pending_id = p["pending_id"]
        promoted_count = p["count"]

        self.pending.pop(pending_idx)

        return mem_id, promoted_pending_id, promoted_count

    def assign(self, emb, global_step, allow_pending=True, allow_update=True):
        """
        Returns:
            mem_id:
                confirmed memory ID, or -1 if still pending/unconfirmed.

            best_sim:
                similarity to best confirmed memory.

            is_new:
                True only when a new confirmed memory speaker is created.

            status:
                one of:
                - matched_confirmed
                - new_pending
                - matched_pending
                - promoted_pending
                - new_confirmed_direct

            pending_id:
                pending candidate ID if relevant, else None.
        """
        emb = l2_normalize_np(emb)

        self.cleanup_pending(global_step)

        best_id, best_sim = self.best_match(emb)

        # 1. Match existing confirmed speaker.
        if best_id is not None and best_sim >= self.threshold:
            if allow_update:
                self.update_confirmed(best_id, emb)

            return {
                "mem_id": best_id,
                "best_sim": best_sim,
                "is_new": False,
                "status": "matched_confirmed",
                "pending_id": None,
                "pending_count": None,
            }

        # 2. If pending is disabled, create confirmed immediately.
        if not allow_pending:
            mem_id = self.add_confirmed(emb, global_step)
            return {
                "mem_id": mem_id,
                "best_sim": best_sim,
                "is_new": True,
                "status": "new_confirmed_direct",
                "pending_id": None,
                "pending_count": None,
            }

        # 3. Try to match an existing pending candidate.
        pending_idx, pending_sim = self.best_pending_match(emb)

        if pending_idx is not None and pending_sim >= self.pending_threshold:
            p = self.update_pending(pending_idx, emb, global_step)

            # Promote if enough hits.
            if p["count"] >= self.pending_confirm_hits:
                mem_id, promoted_pending_id, promoted_count = self.promote_pending(
                    pending_idx,
                    global_step,
                )

                return {
                    "mem_id": mem_id,
                    "best_sim": best_sim,
                    "is_new": True,
                    "status": "promoted_pending",
                    "pending_id": promoted_pending_id,
                    "pending_count": promoted_count,
                }

            return {
                "mem_id": -1,
                "best_sim": best_sim,
                "is_new": False,
                "status": "matched_pending",
                "pending_id": p["pending_id"],
                "pending_count": p["count"],
            }

        # 4. Create new pending candidate, not confirmed speaker.
        pending_id = self.add_pending(emb, global_step)

        return {
            "mem_id": -1,
            "best_sim": best_sim,
            "is_new": False,
            "status": "new_pending",
            "pending_id": pending_id,
            "pending_count": 1,
        }


# ============================================================
# Evaluation helpers
# ============================================================
def cluster_accuracy(pred_labels, true_labels):
    pred = np.array(pred_labels)
    true = np.array(true_labels)

    total = 0
    for c in np.unique(pred):
        idx = pred == c
        subset = true[idx]
        if len(subset) == 0:
            continue
        total += Counter(subset).most_common(1)[0][1]

    return total / max(1, len(true))


def compute_id_switches(true_labels, pred_ids, segment_ids):
    by_spk = defaultdict(list)

    for t, p, seg in zip(true_labels, pred_ids, segment_ids):
        by_spk[t].append((seg, p))

    switches = 0
    total_transitions = 0

    for spk, items in by_spk.items():
        items = sorted(items, key=lambda x: x[0])
        prev = None

        for _, mem_id in items:
            if prev is not None:
                total_transitions += 1
                if mem_id != prev:
                    switches += 1
            prev = mem_id

    switch_rate = switches / max(1, total_transitions)
    return switches, total_transitions, switch_rate


def compute_reid_accuracy(true_labels, pred_ids):
    by_spk = defaultdict(list)

    for t, p in zip(true_labels, pred_ids):
        by_spk[t].append(p)

    correct = 0
    total = 0

    for spk, assigned_ids in by_spk.items():
        if len(assigned_ids) < 2:
            continue

        majority_id = Counter(assigned_ids).most_common(1)[0][0]

        for p in assigned_ids:
            total += 1
            if p == majority_id:
                correct += 1

    return correct / max(1, total)


def parse_active_speakers(row):
    active = str(row["active_speakers"])
    if active.strip() == "" or active == "nan":
        return []
    return active.split("|")


def valid_confirmed_records(records_df):
    """
    Pending assignments have pred_memory_id == -1.
    For final NMI/ARI/ReID, we usually evaluate only confirmed assignments.
    """
    if len(records_df) == 0:
        return records_df
    return records_df[records_df["pred_memory_id"].astype(int) >= 0].copy()


# ============================================================
# Core meeting evaluation
# ============================================================
def evaluate_one_meeting(
    meeting_row,
    dual_model,
    teacher_model,
    threshold=0.55,
    ema_alpha=None,
    pending_threshold=0.60,
    pending_confirm_hits=2,
    pending_max_age=5,
    update_only_single=False,
    device="cuda",
    sr=16000,
    use_teacher_for_single_speaker_selection=True,
):
    meeting_id = meeting_row["meeting_id"]
    mix_path = meeting_row["mix_path"]
    timeline_csv = meeting_row["timeline_csv"]
    speakers_csv = meeting_row["speakers_csv"]

    timeline = pd.read_csv(timeline_csv)
    speakers_df = pd.read_csv(speakers_csv)

    spk_to_ref = {
        str(r["speaker_id"]): r["speaker_ref_path"]
        for _, r in speakers_df.iterrows()
    }

    wav, wav_sr = torchaudio.load(mix_path)
    wav = wav.mean(0)

    if wav_sr != sr:
        wav = torchaudio.functional.resample(wav, wav_sr, sr)

    memory = SpeakerMemoryBank(
        threshold=threshold,
        ema_alpha=ema_alpha,
        pending_threshold=pending_threshold,
        pending_confirm_hits=pending_confirm_hits,
        pending_max_age=pending_max_age,
    )

    records = []
    global_step = 0

    for _, row in timeline.iterrows():
        seg_id = int(row["segment_id"])
        start_sec = float(row["start_sec"])
        end_sec = float(row["end_sec"])
        active_speakers = parse_active_speakers(row)

        if len(active_speakers) == 0:
            continue

        s = int(start_sec * sr)
        e = int(end_sec * sr)
        chunk = wav[s:e]

        if chunk.numel() < int(0.5 * sr):
            continue

        pred_embs = get_dual_embeddings(dual_model, chunk, device=device)

        # ----------------------------------------------------
        # Case 1: one active speaker.
        # Pick best embedding and ignore duplicate/collapsed one.
        # ----------------------------------------------------
        if len(active_speakers) == 1:
            true_spk = str(active_speakers[0])

            selected_idx = 0
            selection_score = None

            if use_teacher_for_single_speaker_selection:
                ref_path = spk_to_ref[true_spk]
                t_emb = get_teacher_emb_from_ref_track(
                    teacher_model,
                    ref_path,
                    start_sec,
                    end_sec,
                    device=device,
                    sr=sr,
                )

                if t_emb is not None:
                    t_np = t_emb.detach().cpu().numpy()
                    sims = [cosine_np(pred_embs[i], t_np) for i in range(2)]
                    selected_idx = int(np.argmax(sims))
                    selection_score = float(np.max(sims))
                else:
                    sims = []
                    for i in range(2):
                        _, sim = memory.best_match(pred_embs[i])
                        sims.append(sim)
                    selected_idx = int(np.argmax(sims))
                    selection_score = float(np.max(sims))
            else:
                if len(memory) > 0:
                    sims = []
                    for i in range(2):
                        _, sim = memory.best_match(pred_embs[i])
                        sims.append(sim)
                    selected_idx = int(np.argmax(sims))
                    selection_score = float(np.max(sims))
                else:
                    selected_idx = 0
                    selection_score = None

            emb = pred_embs[selected_idx]

            assign_out = memory.assign(
                emb,
                global_step,
                allow_pending=True,
                allow_update=True,
            )

            records.append({
                "meeting_id": meeting_id,
                "segment_id": seg_id,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "true_speaker": true_spk,
                "pred_memory_id": assign_out["mem_id"],
                "num_active": 1,
                "selected_embedding_idx": selected_idx,
                "ignored_embedding_idx": 1 - selected_idx,
                "selection_score": selection_score,
                "memory_match_score": assign_out["best_sim"],
                "created_new_memory": assign_out["is_new"],
                "assignment_status": assign_out["status"],
                "pending_id": assign_out["pending_id"],
                "pending_count": assign_out["pending_count"],
            })

            global_step += 1

        # ----------------------------------------------------
        # Case 2: two active speakers.
        # Teacher PIT is used only for evaluation mapping.
        # ----------------------------------------------------
        elif len(active_speakers) == 2:
            spk1 = str(active_speakers[0])
            spk2 = str(active_speakers[1])

            t1 = get_teacher_emb_from_ref_track(
                teacher_model,
                spk_to_ref[spk1],
                start_sec,
                end_sec,
                device=device,
                sr=sr,
            )
            t2 = get_teacher_emb_from_ref_track(
                teacher_model,
                spk_to_ref[spk2],
                start_sec,
                end_sec,
                device=device,
                sr=sr,
            )

            if t1 is None or t2 is None:
                mapped = [
                    (pred_embs[0], spk1, 0, None),
                    (pred_embs[1], spk2, 1, None),
                ]
            else:
                t1_np = t1.detach().cpu().numpy()
                t2_np = t2.detach().cpu().numpy()

                e0 = pred_embs[0]
                e1 = pred_embs[1]

                score_direct = cosine_np(e0, t1_np) + cosine_np(e1, t2_np)
                score_swap = cosine_np(e0, t2_np) + cosine_np(e1, t1_np)

                if score_direct >= score_swap:
                    mapped = [
                        (e0, spk1, 0, cosine_np(e0, t1_np)),
                        (e1, spk2, 1, cosine_np(e1, t2_np)),
                    ]
                else:
                    mapped = [
                        (e0, spk2, 0, cosine_np(e0, t2_np)),
                        (e1, spk1, 1, cosine_np(e1, t1_np)),
                    ]

            for emb, true_spk, emb_idx, sel_score in mapped:
                allow_update = not update_only_single

                assign_out = memory.assign(
                    emb,
                    global_step,
                    allow_pending=True,
                    allow_update=allow_update,
                )

                records.append({
                    "meeting_id": meeting_id,
                    "segment_id": seg_id,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "true_speaker": true_spk,
                    "pred_memory_id": assign_out["mem_id"],
                    "num_active": 2,
                    "selected_embedding_idx": emb_idx,
                    "ignored_embedding_idx": "",
                    "selection_score": sel_score,
                    "memory_match_score": assign_out["best_sim"],
                    "created_new_memory": assign_out["is_new"],
                    "assignment_status": assign_out["status"],
                    "pending_id": assign_out["pending_id"],
                    "pending_count": assign_out["pending_count"],
                })

                global_step += 1

        else:
            raise ValueError(
                f"This evaluator assumes max_overlap <= 2, "
                f"but segment {seg_id} has {len(active_speakers)} active speakers."
            )

    records_df = pd.DataFrame(records)

    if len(records_df) == 0:
        return None, None

    confirmed_df = valid_confirmed_records(records_df)

    if len(confirmed_df) == 0:
        metrics = {
            "meeting_id": meeting_id,
            "num_true_speakers": 0,
            "num_memory_speakers": 0,
            "num_records": len(records_df),
            "num_confirmed_records": 0,
            "confirmation_rate": 0.0,
            "nmi": 0.0,
            "ari": 0.0,
            "cluster_acc": 0.0,
            "id_switches": 0,
            "id_transitions": 0,
            "id_switch_rate": 0.0,
            "reid_accuracy": 0.0,
            "memory_bank_size": len(memory),
            "pending_left": len(memory.pending),
        }
        return metrics, records_df

    true_labels = confirmed_df["true_speaker"].astype(str).values
    pred_ids = confirmed_df["pred_memory_id"].astype(int).values
    segment_ids = confirmed_df["segment_id"].astype(int).values

    nmi = normalized_mutual_info_score(true_labels, pred_ids)
    ari = adjusted_rand_score(true_labels, pred_ids)
    acc = cluster_accuracy(pred_ids, true_labels)

    switches, transitions, switch_rate = compute_id_switches(
        true_labels,
        pred_ids,
        segment_ids,
    )

    reid_acc = compute_reid_accuracy(true_labels, pred_ids)

    metrics = {
        "meeting_id": meeting_id,
        "num_true_speakers": len(np.unique(true_labels)),
        "num_memory_speakers": len(np.unique(pred_ids)),
        "num_records": len(records_df),
        "num_confirmed_records": len(confirmed_df),
        "confirmation_rate": len(confirmed_df) / max(1, len(records_df)),
        "nmi": float(nmi),
        "ari": float(ari),
        "cluster_acc": float(acc),
        "id_switches": int(switches),
        "id_transitions": int(transitions),
        "id_switch_rate": float(switch_rate),
        "reid_accuracy": float(reid_acc),
        "memory_bank_size": len(memory),
        "pending_left": len(memory.pending),
    }

    return metrics, records_df


# ============================================================
# Main
# ============================================================
def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print("Using device:", device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading teacher model...")
    teacher = load_teacher_model(args.teacher_ckpt, device=device)

    print("Loading dual model...")
    dual = load_dual_model(
        args.dual_ckpt,
        joint_ckpt_path=args.joint_dual_ckpt,
        device=device,
    )

    manifest = pd.read_csv(args.manifest_csv)

    all_metrics = []
    all_records = []

    for _, meeting_row in tqdm(
        manifest.iterrows(),
        total=len(manifest),
        desc="Evaluating meetings",
    ):
        metrics, records_df = evaluate_one_meeting(
            meeting_row=meeting_row,
            dual_model=dual,
            teacher_model=teacher,
            threshold=args.threshold,
            ema_alpha=args.ema_alpha,
            pending_threshold=args.pending_threshold,
            pending_confirm_hits=args.pending_confirm_hits,
            pending_max_age=args.pending_max_age,
            update_only_single=args.update_only_single,
            device=device,
            sr=args.sample_rate,
            use_teacher_for_single_speaker_selection=not args.no_teacher_single_selection,
        )

        if metrics is None:
            continue

        all_metrics.append(metrics)
        all_records.append(records_df)

        print(
            f"[{metrics['meeting_id']}] "
            f"NMI={metrics['nmi']:.3f} "
            f"ARI={metrics['ari']:.3f} "
            f"ReID={metrics['reid_accuracy']:.3f} "
            f"SwitchRate={metrics['id_switch_rate']:.3f} "
            f"Memory={metrics['memory_bank_size']} "
            f"TrueSpk={metrics['num_true_speakers']} "
            f"Confirm={metrics['confirmation_rate']:.3f}"
        )

    metrics_df = pd.DataFrame(all_metrics)

    if len(all_records) > 0:
        records_all = pd.concat(all_records, ignore_index=True)
    else:
        records_all = pd.DataFrame()

    metrics_path = out_dir / "meeting_memory_metrics.csv"
    records_path = out_dir / "meeting_memory_assignments.csv"

    metrics_df.to_csv(metrics_path, index=False)
    records_all.to_csv(records_path, index=False)

    print("\n================ FINAL SUMMARY ================")
    print(metrics_df.mean(numeric_only=True))

    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved assignments to: {records_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--dual_ckpt", type=str, required=True)
    parser.add_argument("--joint_dual_ckpt", type=str, default=None)
    parser.add_argument("--teacher_ckpt", type=str, required=True)

    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--ema_alpha", type=float, default=None)

    parser.add_argument("--pending_threshold", type=float, default=0.60)
    parser.add_argument("--pending_confirm_hits", type=int, default=2)
    parser.add_argument("--pending_max_age", type=int, default=5)

    parser.add_argument(
        "--update_only_single",
        action="store_true",
        help="If set, confirmed centroids are only updated by single-speaker chunks.",
    )

    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument(
        "--no_teacher_single_selection",
        action="store_true",
        help="If set, single-speaker chunks choose embedding using memory similarity instead of teacher similarity.",
    )

    args = parser.parse_args()
    main(args)