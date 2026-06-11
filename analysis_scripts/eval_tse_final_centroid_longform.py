#!/usr/bin/env python3
"""Evaluate final-centroid conditioned TSE on long-form conversations."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    import torchaudio
    import yaml
except ImportError as exc:
    raise SystemExit(
        "This script requires torch, torchaudio, and pyyaml in the active environment."
    ) from exc

sys.path.append("/home/sidcs/codebase/wavlm_dual_embedding")
from model import SpeakerEncoderDualWrapper

sys.path.append("/home/sidcs/codebase")
from wavlm_single_embedding.model import SpeakerEncoderWrapper as SingleSpkEncoder

sys.path.append("/home/sidcs/codebase/wesep")
from models.dpcnn import DPCCN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dual-base-ckpt", required=True)
    parser.add_argument("--dual-joint-ckpt", required=True)
    parser.add_argument("--teacher-ckpt", required=True)
    parser.add_argument("--tse-ckpt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--tse-config",
        default="/home/sidcs/codebase/wesep/confs/config_dpcnn.yaml",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--emb-dim", type=int, default=256)
    parser.add_argument("--add-noise", action="store_true")
    parser.add_argument(
        "--noise-dir",
        default="/home/sidcs/datasets/LibriMix/LibriMix/wham_noise/tt",
    )
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--limit-conversations", type=int, default=None)
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


def strip_model_prefix(state):
    new_state = {}
    for key, value in state.items():
        if key.startswith("model."):
            new_state[key.replace("model.", "", 1)] = value
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


def load_tse_model(tse_ckpt: str, tse_config: str, device: str):
    with open(tse_config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    model = DPCCN(**cfg["model_args"]["tse_model"])
    ckpt = torch.load(tse_ckpt, map_location=device)
    model.load_state_dict(strip_model_prefix(ckpt["state_dict"]), strict=True)
    model.to(device).eval()
    return model


def load_audio_mono(path: str) -> torch.Tensor:
    wav, _ = torchaudio.load(path)
    return wav.mean(0)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a @ b) / (a.norm() * b.norm() + 1e-8))


def normalize_centroid(embs: list[torch.Tensor]) -> torch.Tensor:
    return F.normalize(torch.stack(embs, dim=0).mean(dim=0), dim=0)


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> float:
    est = est - est.mean()
    ref = ref - ref.mean()
    ref_energy = torch.sum(ref * ref) + eps
    proj = torch.sum(est * ref) * ref / ref_energy
    noise = est - proj
    ratio = torch.sum(proj * proj) / (torch.sum(noise * noise) + eps)
    return float(10.0 * torch.log10(ratio + eps))


def mix_with_snr(clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    if noise.ndim > 1:
        noise = noise.mean(0)

    if len(noise) < len(clean):
        diff = len(clean) - len(noise)
        noise = F.pad(noise, (diff // 2, diff - diff // 2))
    else:
        noise = noise[: len(clean)]

    clean_power = clean.pow(2).mean()
    noise_power = noise.pow(2).mean()
    target_noise_power = clean_power / (10 ** (snr_db / 10))
    scale = torch.sqrt(target_noise_power / (noise_power + 1e-8))
    return clean + scale * noise


def unwrap_tse_output(output) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def get_teacher_embedding(
    teacher_model,
    wav_path: str,
    device: str,
    cache: dict[str, torch.Tensor],
) -> torch.Tensor:
    if wav_path in cache:
        return cache[wav_path]
    wav = load_audio_mono(wav_path).to(device).unsqueeze(0)
    with torch.no_grad():
        emb = teacher_model(wav).squeeze(0)
    cache[wav_path] = emb.detach().clone()
    return cache[wav_path]


def align_dual_embeddings(
    predicted: torch.Tensor,
    chunk: dict[str, object],
    teacher_embs: list[torch.Tensor],
) -> list[dict[str, object]]:
    direct = cosine(predicted[0], teacher_embs[0]) + cosine(predicted[1], teacher_embs[1])
    swap = cosine(predicted[0], teacher_embs[1]) + cosine(predicted[1], teacher_embs[0])
    order = [0, 1] if direct >= swap else [1, 0]

    aligned = []
    for pred_idx, src_idx in enumerate(order):
        aligned.append(
            {
                "speaker_id": chunk["active_global_speakers"][src_idx],
                "source_path": chunk["source_chunk_paths"][src_idx],
                "dual_emb": predicted[pred_idx].detach().clone(),
            }
        )
    return aligned


def evaluate_conversation(
    conv_dir: Path,
    dual_model,
    teacher_model,
    tse_model,
    device: str,
    add_noise: bool,
    noise_files: list[str],
    rng: random.Random,
) -> dict[str, object]:
    with (conv_dir / "metadata.json").open() as handle:
        metadata = json.load(handle)

    teacher_cache: dict[str, torch.Tensor] = {}
    speaker_bank: dict[str, list[torch.Tensor]] = defaultdict(list)
    noisy_chunks = []

    for chunk in metadata["chunks"]:
        clean_mix = load_audio_mono(chunk["mix_chunk_path"])
        noisy_mix = clean_mix.clone()
        noise_info = None
        if add_noise and noise_files:
            noise_path = rng.choice(noise_files)
            noise_wav = load_audio_mono(noise_path)
            r = rng.random()
            if r < 0.4:
                snr = rng.uniform(-5, 5)
            elif r < 0.8:
                snr = rng.uniform(5, 15)
            else:
                snr = rng.uniform(15, 25)
            noisy_mix = mix_with_snr(noisy_mix, noise_wav, snr)
            noise_info = {"noise_path": noise_path, "snr_db": snr}

        with torch.no_grad():
            predicted = dual_model(noisy_mix.to(device).unsqueeze(0)).squeeze(0)

        teacher_embs = [
            get_teacher_embedding(teacher_model, wav_path, device, teacher_cache)
            for wav_path in chunk["source_chunk_paths"]
        ]
        aligned = align_dual_embeddings(predicted, chunk, teacher_embs)
        for item in aligned:
            speaker_bank[item["speaker_id"]].append(item["dual_emb"])

        noisy_chunks.append(
            {
                "chunk_index": chunk["chunk_index"],
                "mix_wav": noisy_mix,
                "chunk_meta": chunk,
                "aligned": aligned,
                "noise_info": noise_info,
            }
        )

    final_centroids = {
        speaker_id: normalize_centroid(embs) for speaker_id, embs in speaker_bank.items()
    }

    speaker_results = []
    for speaker_id, centroid in final_centroids.items():
        pred_segments = []
        target_segments = []
        mix_segments = []
        active_chunks = 0

        for record in noisy_chunks:
            active = next((item for item in record["aligned"] if item["speaker_id"] == speaker_id), None)
            if active is None:
                continue
            active_chunks += 1

            mix_wav = record["mix_wav"]
            target_wav = load_audio_mono(active["source_path"])

            with torch.no_grad():
                pred_wav = unwrap_tse_output(
                    tse_model(mix_wav.to(device).unsqueeze(0), centroid.to(device).unsqueeze(0))
                ).squeeze(0).cpu()

            min_len = min(len(pred_wav), len(target_wav), len(mix_wav))
            pred_segments.append(pred_wav[:min_len])
            target_segments.append(target_wav[:min_len])
            mix_segments.append(mix_wav[:min_len])

        if not pred_segments:
            continue

        pred_cat = torch.cat(pred_segments, dim=0)
        target_cat = torch.cat(target_segments, dim=0)
        mix_cat = torch.cat(mix_segments, dim=0)

        target_sisdr = si_sdr(pred_cat, target_cat)
        mix_sisdr = si_sdr(mix_cat, target_cat)
        speaker_results.append(
            {
                "speaker_id": speaker_id,
                "num_active_chunks": active_chunks,
                "sisdr": target_sisdr,
                "sisdri": target_sisdr - mix_sisdr,
            }
        )

    return {
        "conversation_dir": str(conv_dir),
        "setting_total_speakers": metadata["setting_total_speakers"],
        "num_chunks": metadata["num_chunks"],
        "num_speakers_evaluated": len(speaker_results),
        "speaker_results": speaker_results,
    }


def aggregate_results(results: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for result in results:
        grouped[int(result["setting_total_speakers"])].append(result)

    summary = {}
    for setting, rows in sorted(grouped.items()):
        flat = [speaker for row in rows for speaker in row["speaker_results"]]
        summary[str(setting)] = {
            "num_conversations": len(rows),
            "num_speaker_evals": len(flat),
            "tse_sisdr_mean": float(np.mean([item["sisdr"] for item in flat])) if flat else None,
            "tse_sisdri_mean": float(np.mean([item["sisdri"] for item in flat])) if flat else None,
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
    tse_model = load_tse_model(args.tse_ckpt, args.tse_config, args.device)

    noise_files = []
    if args.add_noise and os.path.isdir(args.noise_dir):
        noise_files = [
            os.path.join(args.noise_dir, name)
            for name in os.listdir(args.noise_dir)
            if name.endswith(".wav")
        ]

    conv_dirs = sorted(dataset_root.glob("setting_*sp/conversation_*"))
    if args.limit_conversations is not None:
        conv_dirs = conv_dirs[: args.limit_conversations]
    if not conv_dirs:
        raise RuntimeError(f"No conversations found under {dataset_root}")

    all_results = []
    for conv_dir in conv_dirs:
        print(f"[eval] {conv_dir}")
        result = evaluate_conversation(
            conv_dir,
            dual_model,
            teacher_model,
            tse_model,
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
