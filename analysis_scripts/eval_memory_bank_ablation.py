#!/usr/bin/env python3
"""Evaluate a fixed K=2 dual-speaker model with a cosine memory bank."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

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
    parser.add_argument("--threshold", type=float, default=0.65)
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

    # Highest-confidence existing matches first, while enforcing one unique ID per active speaker.
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
    speaker_first_pred: dict[str, int] = {}
    speaker_last_pred: dict[str, int] = {}
    events = []
    reid_total = 0
    reid_correct = 0
    id_switches = 0
    same_speaker_scores = []
    different_speaker_scores = []
    speaker_last_emb: dict[str, torch.Tensor] = {}
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
            else:
                aligned = [
                    (predicted[0], chunk["active_global_speakers"][1]),
                    (predicted[1], chunk["active_global_speakers"][0]),
                ]
        else:
            # In a 1-speaker chunk, the model still emits two embeddings.
            # We intentionally score only the better-aligned embedding and
            # ignore the other one so collapsed duplicate outputs are not penalized.
            score0 = cosine(predicted[0], teacher_embs[0])
            score1 = cosine(predicted[1], teacher_embs[0])
            emb = predicted[0] if score0 >= score1 else predicted[1]
            aligned = [(emb, chunk["active_global_speakers"][0])]
            ignored_extra_predictions += 1

        assigned = assign_chunk_embeddings(aligned, memory_bank, threshold)

        for emb, speaker_id, pred_id, best_score, created_new in assigned:
            if speaker_id in speaker_last_emb:
                same_speaker_scores.append(cosine(emb, speaker_last_emb[speaker_id]))
            for other_speaker, other_emb in speaker_last_emb.items():
                if other_speaker != speaker_id:
                    different_speaker_scores.append(cosine(emb, other_emb))
            speaker_last_emb[speaker_id] = emb.detach().clone()

            seen_before = speaker_id in speaker_first_pred
            if not seen_before:
                speaker_first_pred[speaker_id] = pred_id
            else:
                reid_total += 1
                if pred_id == speaker_first_pred[speaker_id]:
                    reid_correct += 1
                if speaker_last_pred[speaker_id] != pred_id:
                    id_switches += 1
            speaker_last_pred[speaker_id] = pred_id
            events.append(
                {
                    "speaker_id": speaker_id,
                    "pred_id": pred_id,
                    "best_score": best_score,
                    "created_new_id": created_new,
                    "chunk_index": chunk["chunk_index"],
                }
            )

    gt = [event["speaker_id"] for event in events]
    pred = [event["pred_id"] for event in events]
    ari = adjusted_rand_score(gt, pred) if len(set(gt)) > 1 else 1.0
    first_id_reid_acc = reid_correct / reid_total if reid_total > 0 else 1.0

    by_speaker: dict[str, list[int]] = defaultdict(list)
    for event in events:
        by_speaker[event["speaker_id"]].append(event["pred_id"])

    majority_total = 0
    majority_correct = 0
    for pred_ids in by_speaker.values():
        if not pred_ids:
            continue
        majority_count = Counter(pred_ids).most_common(1)[0][1]
        majority_total += len(pred_ids) - 1
        majority_correct += majority_count - 1
    majority_id_acc = majority_correct / majority_total if majority_total > 0 else 1.0

    return {
        "conversation_dir": str(conv_dir),
        "setting_total_speakers": metadata["setting_total_speakers"],
        "num_chunks": metadata["num_chunks"],
        "num_events": len(events),
        "num_memory_ids": len(memory_bank),
        "reid_accuracy": first_id_reid_acc,
        "first_id_reid_accuracy": first_id_reid_acc,
        "majority_id_accuracy": majority_id_acc,
        "ari": ari,
        "id_switches": id_switches,
        "threshold": threshold,
        "ignored_extra_predictions": ignored_extra_predictions,
        "same_speaker_cosine_mean": float(np.mean(same_speaker_scores)) if same_speaker_scores else None,
        "different_speaker_cosine_mean": float(np.mean(different_speaker_scores)) if different_speaker_scores else None,
        "events": events,
    }


def aggregate_results(results: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for result in results:
        grouped[int(result["setting_total_speakers"])].append(result)

    summary = {}
    for setting, rows in sorted(grouped.items()):
        summary[str(setting)] = {
            "reid_accuracy_mean": float(np.mean([row["reid_accuracy"] for row in rows])),
            "first_id_reid_accuracy_mean": float(
                np.mean([row["first_id_reid_accuracy"] for row in rows])
            ),
            "majority_id_accuracy_mean": float(
                np.mean([row["majority_id_accuracy"] for row in rows])
            ),
            "ari_mean": float(np.mean([row["ari"] for row in rows])),
            "id_switches_mean": float(np.mean([row["id_switches"] for row in rows])),
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
        print(f"[eval] {conv_dir}")
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
    print(f"[done] wrote results to {out_dir}")


if __name__ == "__main__":
    main()
