#!/usr/bin/env python3
"""Evaluate memory-bank centroids against teacher single-speaker centroids."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - fallback when scipy is unavailable
    linear_sum_assignment = None

try:
    import torch
    import torchaudio
except ImportError as exc:
    raise SystemExit(
        "This script requires torch and torchaudio in the active Python environment."
    ) from exc

sys.path.append("/home/sidcs/codebase/wavlm_dual_embedding")
from model import SpeakerEncoderDualWrapper

sys.path.append("/home/sidcs/codebase")
from wavlm_single_embedding.model import SpeakerEncoderWrapper as SingleSpkEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dual-base-ckpt", required=True)
    parser.add_argument("--dual-joint-ckpt", required=True)
    parser.add_argument("--teacher-ckpt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--emb-dim", type=int, default=256)
    parser.add_argument("--add-noise", action="store_true")
    parser.add_argument(
        "--noise-dir",
        default="/home/sidcs/datasets/LibriMix/LibriMix/wham_noise/tt",
    )
    parser.add_argument("--seed", type=int, default=44)
    return parser.parse_args()


def strip_dual_model_weights(state):
    new_state = {}
    for key, value in state.items():
        if not key.startswith("model."):
            continue
        clean_key = key.replace("model.", "")
        if clean_key.startswith("single_sp_model.") or clean_key.startswith("arcface_loss."):
            continue
        new_state[clean_key] = value
    return new_state


def joint_trained_model_weights(state):
    new_state = {}
    for key, value in state.items():
        if key.startswith("dual_emb_model."):
            new_state[key.replace("dual_emb_model.", "")] = value
    return new_state


def load_dual_model(base_ckpt: str, joint_ckpt: str, emb_dim: int, device: str):
    base = torch.load(base_ckpt, map_location=device)
    state = strip_dual_model_weights(base["state_dict"])
    model = SpeakerEncoderDualWrapper(emb_dim=emb_dim)
    model.load_state_dict(state, strict=True)

    joint = torch.load(joint_ckpt, map_location=device)
    model.load_state_dict(joint_trained_model_weights(joint["state_dict"]), strict=True)
    model.to(device).eval()
    return model


def load_teacher_model(teacher_ckpt: str, device: str):
    teacher = SingleSpkEncoder().to(device)
    ckpt = torch.load(teacher_ckpt, map_location=device)
    filtered = {}
    for key, value in ckpt["state_dict"].items():
        if not key.startswith("model."):
            continue
        if "arcface" in key or "arc_face" in key:
            continue
        filtered[key.replace("model.", "", 1)] = value
    teacher.load_state_dict(filtered, strict=True)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a @ b) / (a.norm() * b.norm() + 1e-8))


def mix_with_snr(clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    if noise.ndim > 1:
        noise = noise.mean(0)

    if len(noise) < len(clean):
        diff = len(clean) - len(noise)
        noise = torch.nn.functional.pad(noise, (diff // 2, diff - diff // 2))
    else:
        noise = noise[: len(clean)]

    clean_power = clean.pow(2).mean()
    noise_power = noise.pow(2).mean()
    target_noise_power = clean_power / (10 ** (snr_db / 10))
    scale = torch.sqrt(target_noise_power / (noise_power + 1e-8))
    return clean + scale * noise


def get_teacher_embedding(teacher, wav_path: str, device: str) -> torch.Tensor:
    wav, _ = torchaudio.load(wav_path)
    wav = wav.mean(0).to(device).unsqueeze(0)
    with torch.no_grad():
        emb = teacher(wav)
    return emb.squeeze(0)


def normalize_mean(embs: list[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(embs, dim=0)
    mean = stacked.mean(dim=0)
    return torch.nn.functional.normalize(mean, dim=0)


def assign_with_memory_bank(
    emb: torch.Tensor,
    memory_bank: dict[int, dict[str, torch.Tensor | int]],
    threshold: float,
    blocked_ids: set[int] | None = None,
) -> tuple[int, float, bool]:
    blocked_ids = blocked_ids or set()
    if not memory_bank:
        memory_bank[0] = {"prototype": emb.detach().clone(), "count": 1}
        return 0, 1.0, True

    best_id = None
    best_score = -1.0
    for pred_id, entry in memory_bank.items():
        if pred_id in blocked_ids:
            continue
        score = cosine(emb, entry["prototype"])
        if score > best_score:
            best_score = score
            best_id = pred_id

    if best_score < threshold or best_id is None:
        new_id = max(memory_bank.keys()) + 1
        memory_bank[new_id] = {"prototype": emb.detach().clone(), "count": 1}
        return new_id, best_score, True

    entry = memory_bank[best_id]
    count = int(entry["count"])
    proto = entry["prototype"]
    updated = (proto * count + emb) / (count + 1)
    entry["prototype"] = torch.nn.functional.normalize(updated, dim=0)
    entry["count"] = count + 1
    return best_id, best_score, False


def assign_chunk_embeddings(
    aligned: list[tuple[torch.Tensor, str]],
    memory_bank: dict[int, dict[str, torch.Tensor | int]],
    threshold: float,
) -> list[tuple[torch.Tensor, str, int, float, bool]]:
    if len(aligned) <= 1:
        emb, speaker_id = aligned[0]
        pred_id, best_score, created_new = assign_with_memory_bank(emb, memory_bank, threshold)
        return [(emb, speaker_id, pred_id, best_score, created_new)]

    candidates = []
    for idx, (emb, speaker_id) in enumerate(aligned):
        best_existing_id = None
        best_existing_score = -1.0
        for pred_id, entry in memory_bank.items():
            score = cosine(emb, entry["prototype"])
            if score > best_existing_score:
                best_existing_score = score
                best_existing_id = pred_id
        candidates.append(
            {
                "idx": idx,
                "speaker_id": speaker_id,
                "emb": emb,
                "best_existing_id": best_existing_id,
                "best_existing_score": best_existing_score,
            }
        )

    used_ids: set[int] = set()
    results: list[tuple[torch.Tensor, str, int, float, bool] | None] = [None] * len(aligned)

    for item in sorted(candidates, key=lambda x: x["best_existing_score"], reverse=True):
        idx = item["idx"]
        emb = item["emb"]
        speaker_id = item["speaker_id"]
        pred_id = item["best_existing_id"]
        best_score = item["best_existing_score"]

        if pred_id is not None and best_score >= threshold and pred_id not in used_ids:
            entry = memory_bank[pred_id]
            count = int(entry["count"])
            proto = entry["prototype"]
            updated = (proto * count + emb) / (count + 1)
            entry["prototype"] = torch.nn.functional.normalize(updated, dim=0)
            entry["count"] = count + 1
            used_ids.add(pred_id)
            results[idx] = (emb, speaker_id, pred_id, best_score, False)

    for item in candidates:
        idx = item["idx"]
        if results[idx] is not None:
            continue
        emb = item["emb"]
        speaker_id = item["speaker_id"]
        pred_id, best_score, created_new = assign_with_memory_bank(
            emb,
            memory_bank,
            threshold,
            blocked_ids=used_ids,
        )
        used_ids.add(pred_id)
        results[idx] = (emb, speaker_id, pred_id, best_score, created_new)

    return results


def fallback_one_to_one_mapping(sim_matrix: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    used_cols: set[int] = set()
    flat = []
    for row_idx in range(sim_matrix.shape[0]):
        for col_idx in range(sim_matrix.shape[1]):
            flat.append((sim_matrix[row_idx, col_idx], row_idx, col_idx))
    flat.sort(reverse=True)
    for _, row_idx, col_idx in flat:
        if row_idx in mapping or col_idx in used_cols:
            continue
        mapping[row_idx] = col_idx
        used_cols.add(col_idx)
    return mapping


def evaluate_conversation(
    conv_dir: Path,
    dual_model,
    teacher_model,
    threshold: float,
    device: str,
    add_noise: bool,
    noise_files: list[str],
    rng: random.Random,
) -> dict[str, object]:
    with (conv_dir / "metadata.json").open() as handle:
        metadata = json.load(handle)
    long_mix, _ = torchaudio.load(conv_dir / "mix.wav")
    long_mix = long_mix.mean(0)

    memory_bank: dict[int, dict[str, torch.Tensor | int]] = {}
    events = []
    pred_embs_by_id: dict[int, list[torch.Tensor]] = defaultdict(list)
    teacher_embs_by_speaker: dict[str, list[torch.Tensor]] = defaultdict(list)
    ignored_extra_predictions = 0

    for chunk in metadata["chunks"]:
        mix_wav = long_mix[chunk["start_sample"] : chunk["end_sample"]].clone()
        if add_noise and noise_files:
            noise_wav, _ = torchaudio.load(rng.choice(noise_files))
            noise_wav = noise_wav.mean(0)
            r = rng.random()
            if r < 0.4:
                snr = rng.uniform(-5, 5)
            elif r < 0.8:
                snr = rng.uniform(5, 15)
            else:
                snr = rng.uniform(15, 25)
            mix_wav = mix_with_snr(mix_wav, noise_wav, snr)

        mix_wav = mix_wav.to(device).unsqueeze(0)
        with torch.no_grad():
            predicted = dual_model(mix_wav).squeeze(0)

        teacher_embs = [
            get_teacher_embedding(teacher_model, wav_path, device)
            for wav_path in chunk["source_chunk_paths"]
        ]

        if chunk["num_active_speakers"] == 2:
            direct = cosine(predicted[0], teacher_embs[0]) + cosine(predicted[1], teacher_embs[1])
            swap = cosine(predicted[0], teacher_embs[1]) + cosine(predicted[1], teacher_embs[0])
            if direct >= swap:
                aligned = [
                    (predicted[0], chunk["active_global_speakers"][0]),
                    (predicted[1], chunk["active_global_speakers"][1]),
                ]
                teacher_aligned = [
                    (chunk["active_global_speakers"][0], teacher_embs[0]),
                    (chunk["active_global_speakers"][1], teacher_embs[1]),
                ]
            else:
                aligned = [
                    (predicted[0], chunk["active_global_speakers"][1]),
                    (predicted[1], chunk["active_global_speakers"][0]),
                ]
                teacher_aligned = [
                    (chunk["active_global_speakers"][1], teacher_embs[1]),
                    (chunk["active_global_speakers"][0], teacher_embs[0]),
                ]
        else:
            score0 = cosine(predicted[0], teacher_embs[0])
            score1 = cosine(predicted[1], teacher_embs[0])
            emb = predicted[0] if score0 >= score1 else predicted[1]
            aligned = [(emb, chunk["active_global_speakers"][0])]
            teacher_aligned = [(chunk["active_global_speakers"][0], teacher_embs[0])]
            ignored_extra_predictions += 1

        for speaker_id, teacher_emb in teacher_aligned:
            teacher_embs_by_speaker[speaker_id].append(teacher_emb.detach().clone())

        assigned = assign_chunk_embeddings(aligned, memory_bank, threshold)
        for emb, speaker_id, pred_id, best_score, created_new in assigned:
            pred_embs_by_id[pred_id].append(emb.detach().clone())
            events.append(
                {
                    "speaker_id": speaker_id,
                    "pred_id": pred_id,
                    "best_score": best_score,
                    "created_new_id": created_new,
                    "chunk_index": chunk["chunk_index"],
                }
            )

    pred_ids = sorted(pred_embs_by_id.keys())
    speaker_ids = sorted(teacher_embs_by_speaker.keys())
    pred_centroids = {pred_id: normalize_mean(pred_embs_by_id[pred_id]) for pred_id in pred_ids}
    teacher_centroids = {
        speaker_id: normalize_mean(teacher_embs_by_speaker[speaker_id]) for speaker_id in speaker_ids
    }

    sim_matrix = np.zeros((len(pred_ids), len(speaker_ids)), dtype=np.float32)
    for row_idx, pred_id in enumerate(pred_ids):
        for col_idx, speaker_id in enumerate(speaker_ids):
            sim_matrix[row_idx, col_idx] = cosine(pred_centroids[pred_id], teacher_centroids[speaker_id])

    top1_mapping = {
        pred_ids[row_idx]: speaker_ids[int(np.argmax(sim_matrix[row_idx]))]
        for row_idx in range(len(pred_ids))
    }

    top1_correct = sum(
        int(event["speaker_id"] == top1_mapping[event["pred_id"]]) for event in events
    )
    top1_event_acc = top1_correct / len(events) if events else 1.0

    if linear_sum_assignment is not None:
        row_ind, col_ind = linear_sum_assignment(-sim_matrix)
        one_to_one = {pred_ids[row]: speaker_ids[col] for row, col in zip(row_ind, col_ind)}
    else:
        fallback = fallback_one_to_one_mapping(sim_matrix)
        one_to_one = {pred_ids[row]: speaker_ids[col] for row, col in fallback.items()}

    hungarian_mapping = dict(one_to_one)
    for pred_id in pred_ids:
        if pred_id not in hungarian_mapping:
            row_idx = pred_ids.index(pred_id)
            hungarian_mapping[pred_id] = speaker_ids[int(np.argmax(sim_matrix[row_idx]))]

    hungarian_correct = sum(
        int(event["speaker_id"] == hungarian_mapping[event["pred_id"]]) for event in events
    )
    hungarian_event_acc = hungarian_correct / len(events) if events else 1.0

    same_scores = []
    diff_scores = []
    pred_cluster_majority = {}
    by_pred_id: dict[int, list[str]] = defaultdict(list)
    for event in events:
        by_pred_id[event["pred_id"]].append(event["speaker_id"])
    for pred_id in pred_ids:
        dominant_speaker = Counter(by_pred_id[pred_id]).most_common(1)[0][0]
        pred_cluster_majority[pred_id] = dominant_speaker
        same_scores.append(cosine(pred_centroids[pred_id], teacher_centroids[dominant_speaker]))
        for other_speaker in speaker_ids:
            if other_speaker != dominant_speaker:
                diff_scores.append(cosine(pred_centroids[pred_id], teacher_centroids[other_speaker]))

    return {
        "conversation_dir": str(conv_dir),
        "setting_total_speakers": metadata["setting_total_speakers"],
        "num_chunks": metadata["num_chunks"],
        "num_events": len(events),
        "num_memory_ids": len(memory_bank),
        "num_teacher_speakers": len(teacher_centroids),
        "threshold": threshold,
        "ignored_extra_predictions": ignored_extra_predictions,
        "teacher_centroid_top1_event_accuracy": float(top1_event_acc),
        "teacher_centroid_hungarian_event_accuracy": float(hungarian_event_acc),
        "teacher_centroid_same_speaker_cosine_mean": float(np.mean(same_scores)) if same_scores else None,
        "teacher_centroid_different_speaker_cosine_mean": (
            float(np.mean(diff_scores)) if diff_scores else None
        ),
        "pred_cluster_majority_speakers": {str(k): v for k, v in pred_cluster_majority.items()},
    }


def aggregate_results(results: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for result in results:
        grouped[int(result["setting_total_speakers"])].append(result)

    summary = {}
    for setting, rows in sorted(grouped.items()):
        summary[str(setting)] = {
            "teacher_centroid_top1_event_accuracy_mean": float(
                np.mean([row["teacher_centroid_top1_event_accuracy"] for row in rows])
            ),
            "teacher_centroid_hungarian_event_accuracy_mean": float(
                np.mean([row["teacher_centroid_hungarian_event_accuracy"] for row in rows])
            ),
            "teacher_centroid_same_speaker_cosine_mean": float(
                np.mean([row["teacher_centroid_same_speaker_cosine_mean"] for row in rows])
            ),
            "teacher_centroid_different_speaker_cosine_mean": float(
                np.mean([row["teacher_centroid_different_speaker_cosine_mean"] for row in rows])
            ),
            "ignored_extra_predictions_mean": float(
                np.mean([row["ignored_extra_predictions"] for row in rows])
            ),
            "num_conversations": len(rows),
        }
    return summary


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    dual_model = load_dual_model(args.dual_base_ckpt, args.dual_joint_ckpt, args.emb_dim, args.device)
    teacher_model = load_teacher_model(args.teacher_ckpt, args.device)
    noise_files = []
    if args.add_noise and os.path.isdir(args.noise_dir):
        noise_files = [
            os.path.join(args.noise_dir, name)
            for name in os.listdir(args.noise_dir)
            if name.endswith(".wav")
        ]

    conv_dirs = sorted(dataset_root.glob("setting_*sp/conversation_*"))
    if not conv_dirs:
        raise RuntimeError(f"No conversations found under {dataset_root}")

    all_results = []
    for conv_dir in conv_dirs:
        print(f"[eval-centroids] {conv_dir}")
        result = evaluate_conversation(
            conv_dir,
            dual_model,
            teacher_model,
            args.threshold,
            args.device,
            args.add_noise,
            noise_files,
            rng,
        )
        all_results.append(result)

    summary = aggregate_results(all_results)

    with (out_dir / "conversation_results.json").open("w") as handle:
        json.dump(all_results, handle, indent=2)
    with (out_dir / "setting_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[done] wrote centroid results to {out_dir}")


if __name__ == "__main__":
    main()
