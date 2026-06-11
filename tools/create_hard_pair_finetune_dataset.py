#!/usr/bin/env python3
"""Build teacher centroids and hard-pair noisy 2-speaker mixes for finetuning."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DUAL_ROOT = SCRIPT_DIR.parent
CODEBASE_ROOT = DUAL_ROOT.parent
if str(CODEBASE_ROOT) not in sys.path:
    sys.path.append(str(CODEBASE_ROOT))

from wavlm_single_embedding.model import SpeakerEncoderWrapper  # noqa: E402


RATE = 16000
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--librispeech-root",
        default="/home/sidcs/datasets/LibriMix/LibriMix/LibriSpeech",
    )
    parser.add_argument(
        "--speakers-txt",
        default="/home/sidcs/datasets/LibriMix/LibriMix/LibriSpeech/SPEAKERS.TXT",
    )
    parser.add_argument(
        "--teacher-ckpt",
        default="/home/sidcs/model_ckpts/librispeech_asp_wavlm_tr360/best-epoch=62-val_separation=0.000.ckpt",
    )
    parser.add_argument(
        "--reuse-centroid-pt",
        default=None,
    )
    parser.add_argument(
        "--output-root",
        default="/home/sidcs/datasets/LibriMix/LibriMix/hard_easy_pairs_teacher_centroid",
    )
    parser.add_argument(
        "--train-subsets",
        nargs="+",
        default=["train-clean-360"],
    )
    parser.add_argument(
        "--val-subsets",
        nargs="+",
        default=["dev-clean"],
    )
    parser.add_argument(
        "--centroid-subsets",
        nargs="+",
        default=None,
    )
    parser.add_argument("--segment-sec", type=float, default=5.0)
    parser.add_argument("--min-overlap", type=float, default=0.6)
    parser.add_argument("--max-overlap", type=float, default=1.0)
    parser.add_argument("--speaker-gain-db", type=float, default=-20.0)
    parser.add_argument("--pairs-per-speaker", type=int, default=5)
    parser.add_argument("--examples-per-pair", type=int, default=3)
    parser.add_argument("--hard-pair-ratio", type=int, default=2)
    parser.add_argument("--easy-pair-ratio", type=int, default=1)
    parser.add_argument("--single-examples-per-speaker", type=int, default=0)
    parser.add_argument("--single-val-examples-per-speaker", type=int, default=0)
    parser.add_argument("--max-train-pairs", type=int, default=None)
    parser.add_argument("--max-val-pairs", type=int, default=200)
    parser.add_argument("--min-speaker-utts", type=int, default=5)
    parser.add_argument("--min-centroid-sim", type=float, default=None)
    parser.add_argument(
        "--train-noise-json",
        default="/home/sidcs/datasets/LibriMix/LibriMix/noise_files_embedding_model/freesound_noise_bins.json",
    )
    parser.add_argument(
        "--val-noise-json",
        default="/home/sidcs/datasets/LibriMix/LibriMix/noise_files_embedding_model/wham_tt_noise_bins.json",
    )
    parser.add_argument("--snr-min-db", type=float, default=-5.0)
    parser.add_argument("--snr-max-db", type=float, default=20.0)
    parser.add_argument("--emb-dim", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def choose_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device_arg


def strip_teacher_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    filtered = {}
    for key, value in state_dict.items():
        if key.startswith("model.") and "arcface" not in key:
            filtered[key.replace("model.", "", 1)] = value
    return filtered


def load_teacher_model(ckpt_path: Path, emb_dim: int, device: str) -> SpeakerEncoderWrapper:
    model = SpeakerEncoderWrapper(emb_dim=emb_dim)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(strip_teacher_state(checkpoint["state_dict"]), strict=True)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def parse_speakers_txt(path: Path) -> dict[str, dict[str, object]]:
    metadata: dict[str, dict[str, object]] = {}
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith(";"):
                continue
            parts = [part.strip() for part in line.split("|")]
            if len(parts) != 5 or parts[0] == "ID":
                continue
            speaker_id, sex, subset, minutes, name = parts
            metadata[speaker_id] = {
                "speaker_id": speaker_id,
                "sex": sex,
                "subset": subset,
                "minutes": float(minutes),
                "name": name,
            }
    return metadata


def load_audio(audio_path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(audio_path))
    if wav.dim() == 2:
        wav = wav.mean(dim=0)
    else:
        wav = wav.squeeze(0)
    if sr != RATE:
        wav = torchaudio.functional.resample(wav, sr, RATE)
    return wav.float()


def normalize_embedding(embedding: torch.Tensor) -> torch.Tensor:
    return F.normalize(embedding, dim=-1, eps=EPS)


def embed_waveform(model: SpeakerEncoderWrapper, wav: torch.Tensor, device: str) -> torch.Tensor:
    with torch.no_grad():
        emb = model(wav.unsqueeze(0).to(device)).squeeze(0).detach().cpu()
    return normalize_embedding(emb)


def discover_speakers(
    librispeech_root: Path,
    speakers_meta: dict[str, dict[str, object]],
    min_speaker_utts: int,
    subset_filter: set[str] | None,
) -> dict[str, dict[str, object]]:
    discovered: dict[str, dict[str, object]] = {}
    for subset_dir in sorted(path for path in librispeech_root.iterdir() if path.is_dir()):
        subset = subset_dir.name
        if subset_filter is not None and subset not in subset_filter:
            continue
        for speaker_dir in sorted(path for path in subset_dir.iterdir() if path.is_dir()):
            speaker_id = speaker_dir.name
            utts = sorted(speaker_dir.rglob("*.flac"))
            if len(utts) < min_speaker_utts:
                continue
            meta = dict(speakers_meta.get(speaker_id, {}))
            meta.update(
                {
                    "speaker_id": speaker_id,
                    "subset": meta.get("subset", subset),
                    "sex": meta.get("sex", "U"),
                    "name": meta.get("name", speaker_id),
                    "minutes": meta.get("minutes"),
                    "utterance_paths": [str(path) for path in utts],
                    "num_utterances": len(utts),
                }
            )
            discovered[speaker_id] = meta
    return discovered


def compute_speaker_centroids(
    teacher_model: SpeakerEncoderWrapper,
    speakers: dict[str, dict[str, object]],
    device: str,
) -> dict[str, torch.Tensor]:
    centroids: dict[str, torch.Tensor] = {}
    iterator = tqdm(sorted(speakers), desc="Computing teacher centroids")
    for speaker_id in iterator:
        utt_paths = speakers[speaker_id]["utterance_paths"]
        utt_embs = []
        for utt_path in utt_paths:
            utt_embs.append(embed_waveform(teacher_model, load_audio(Path(utt_path)), device))
        centroid = normalize_embedding(torch.stack(utt_embs, dim=0).mean(dim=0))
        centroids[speaker_id] = centroid
    return centroids


def write_centroid_metadata(
    output_root: Path,
    speakers: dict[str, dict[str, object]],
    centroids: dict[str, torch.Tensor],
) -> tuple[Path, Path]:
    centroid_dir = output_root / "centroids"
    centroid_dir.mkdir(parents=True, exist_ok=True)

    centroid_pt = centroid_dir / "teacher_centroids.pt"
    speaker_ids = sorted(centroids)
    centroid_matrix = torch.stack([centroids[speaker_id] for speaker_id in speaker_ids], dim=0)
    torch.save(
        {
            "speaker_ids": speaker_ids,
            "embeddings": centroid_matrix,
            "metadata": {speaker_id: speakers[speaker_id] for speaker_id in speaker_ids},
        },
        centroid_pt,
    )

    centroid_csv = centroid_dir / "teacher_centroids.csv"
    with centroid_csv.open("w", newline="") as handle:
        fieldnames = [
            "speaker_id",
            "subset",
            "sex",
            "name",
            "num_utterances",
            "embedding_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for speaker_id in speaker_ids:
            row = speakers[speaker_id]
            writer.writerow(
                {
                    "speaker_id": speaker_id,
                    "subset": row["subset"],
                    "sex": row["sex"],
                    "name": row["name"],
                    "num_utterances": row["num_utterances"],
                    "embedding_path": str(centroid_pt),
                }
            )
    return centroid_pt, centroid_csv


def load_centroid_metadata(centroid_pt: Path) -> tuple[dict[str, dict[str, object]], dict[str, torch.Tensor]]:
    payload = torch.load(centroid_pt, map_location="cpu")
    speaker_ids = payload["speaker_ids"]
    embeddings = payload["embeddings"]
    metadata = payload["metadata"]
    speakers = {speaker_id: metadata[speaker_id] for speaker_id in speaker_ids}
    centroids = {
        speaker_id: normalize_embedding(embeddings[idx].float())
        for idx, speaker_id in enumerate(speaker_ids)
    }
    return speakers, centroids


def build_similarity_pairs(
    speakers: dict[str, dict[str, object]],
    centroids: dict[str, torch.Tensor],
    subsets: list[str],
    pairs_per_speaker: int,
    max_pairs: int | None,
    min_similarity: float | None,
    selection_mode: str = "hard",
) -> list[dict[str, object]]:
    eligible = [
        speaker_id
        for speaker_id, meta in speakers.items()
        if meta["subset"] in subsets and speaker_id in centroids
    ]
    eligible = sorted(eligible)
    if len(eligible) < 2:
        return []

    mat = torch.stack([centroids[speaker_id] for speaker_id in eligible], dim=0)
    sims = mat @ mat.T

    unique_pairs: dict[tuple[str, str], dict[str, object]] = {}
    for i, speaker_id in enumerate(eligible):
        values, indices = torch.sort(sims[i], descending=True)
        chosen = 0
        for sim_value, j in zip(values.tolist(), indices.tolist()):
            if i == j:
                continue
            if min_similarity is not None and sim_value < min_similarity:
                break
            other_id = eligible[j]
            pair_key = tuple(sorted((speaker_id, other_id)))
            if pair_key not in unique_pairs or sim_value > unique_pairs[pair_key]["pair_cosine"]:
                unique_pairs[pair_key] = {
                    "speaker_1_ID": pair_key[0],
                    "speaker_2_ID": pair_key[1],
                    "pair_cosine": float(sim_value),
                    "speaker_1_subset": speakers[pair_key[0]]["subset"],
                    "speaker_2_subset": speakers[pair_key[1]]["subset"],
                }
            chosen += 1
            if chosen >= pairs_per_speaker:
                break

    ranked = sorted(unique_pairs.values(), key=lambda item: item["pair_cosine"], reverse=True)
    if selection_mode == "easy":
        ranked = list(reversed(ranked))
    if max_pairs is not None:
        ranked = ranked[:max_pairs]
    return ranked


def build_ranked_pair_pool(
    speakers: dict[str, dict[str, object]],
    centroids: dict[str, torch.Tensor],
    subsets: list[str],
    min_similarity: float | None = None,
) -> list[dict[str, object]]:
    eligible = [
        speaker_id
        for speaker_id, meta in speakers.items()
        if meta["subset"] in subsets and speaker_id in centroids
    ]
    eligible = sorted(eligible)
    if len(eligible) < 2:
        return []

    rows = []
    for i, speaker_1_id in enumerate(eligible):
        c1 = centroids[speaker_1_id]
        for j in range(i + 1, len(eligible)):
            speaker_2_id = eligible[j]
            c2 = centroids[speaker_2_id]
            pair_cosine = float(torch.dot(c1, c2))
            if min_similarity is not None and pair_cosine < min_similarity:
                continue
            rows.append(
                {
                    "speaker_1_ID": speaker_1_id,
                    "speaker_2_ID": speaker_2_id,
                    "pair_cosine": pair_cosine,
                    "speaker_1_subset": speakers[speaker_1_id]["subset"],
                    "speaker_2_subset": speakers[speaker_2_id]["subset"],
                }
            )

    return sorted(rows, key=lambda item: item["pair_cosine"], reverse=True)


def select_hard_easy_pairs(
    speakers: dict[str, dict[str, object]],
    centroids: dict[str, torch.Tensor],
    subsets: list[str],
    pairs_per_speaker: int,
    max_pairs: int | None,
    min_similarity: float | None,
    hard_ratio: int,
    easy_ratio: int,
) -> list[dict[str, object]]:
    pair_pool = build_ranked_pair_pool(
        speakers=speakers,
        centroids=centroids,
        subsets=subsets,
        min_similarity=min_similarity,
    )
    if not pair_pool:
        return []

    total_target_pairs = max_pairs
    if total_target_pairs is None:
        eligible_count = sum(int(meta["subset"] in subsets) for meta in speakers.values())
        total_target_pairs = max(1, eligible_count * pairs_per_speaker)
    total_target_pairs = min(total_target_pairs, len(pair_pool))

    ratio_sum = hard_ratio + easy_ratio
    hard_target = int(round(total_target_pairs * (hard_ratio / ratio_sum)))
    hard_target = min(max(hard_target, 1), total_target_pairs)
    easy_target = total_target_pairs - hard_target

    selected = []
    seen = set()

    for pair in pair_pool:
        if len(selected) >= hard_target:
            break
        key = (pair["speaker_1_ID"], pair["speaker_2_ID"])
        if key in seen:
            continue
        selected_pair = dict(pair)
        selected_pair["pair_difficulty"] = "hard"
        selected.append(selected_pair)
        seen.add(key)

    if easy_target > 0:
        for pair in reversed(pair_pool):
            if len(selected) >= total_target_pairs:
                break
            key = (pair["speaker_1_ID"], pair["speaker_2_ID"])
            if key in seen:
                continue
            selected_pair = dict(pair)
            selected_pair["pair_difficulty"] = "easy"
            selected.append(selected_pair)
            seen.add(key)

    return selected


def choose_noise_file(noise_bins: dict[str, list[str]], duration_sec: float, rng: random.Random) -> str:
    bins = []
    for key, paths in noise_bins.items():
        if not paths:
            continue
        lower, upper = key.split("-")
        bins.append((int(lower), int(upper), key))
    if not bins:
        raise RuntimeError("No non-empty noise bins were found.")
    bins.sort(key=lambda item: item[1])

    for _, upper, key in bins:
        if duration_sec <= upper:
            return rng.choice(noise_bins[key])
    return rng.choice(noise_bins[bins[-1][2]])


def rms_normalize(wav: torch.Tensor, gain_db: float) -> torch.Tensor:
    rms = wav.pow(2).mean().sqrt().clamp_min(EPS)
    target = math.pow(10.0, gain_db / 20.0)
    return wav * (target / rms)


def mix_noise_with_snr(clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    if noise.ndim > 1:
        noise = noise.mean(dim=0)
    if len(noise) < len(clean):
        repeats = math.ceil(len(clean) / max(len(noise), 1))
        noise = noise.repeat(repeats)
    noise = noise[: len(clean)]
    clean_power = clean.pow(2).mean()
    noise_power = noise.pow(2).mean().clamp_min(EPS)
    target_noise_power = clean_power / math.pow(10.0, snr_db / 10.0)
    scale = torch.sqrt(target_noise_power / noise_power)
    return clean + scale * noise


def random_crop_or_pad(wav: torch.Tensor, target_len: int, rng: random.Random) -> torch.Tensor:
    if len(wav) >= target_len:
        max_start = len(wav) - target_len
        start = rng.randint(0, max_start) if max_start > 0 else 0
        return wav[start : start + target_len].clone()
    pad_len = target_len - len(wav)
    left = rng.randint(0, pad_len) if pad_len > 0 else 0
    right = pad_len - left
    return F.pad(wav, (left, right))


def peak_normalize(*wavs: torch.Tensor) -> list[torch.Tensor]:
    peak = max(float(wav.abs().max()) for wav in wavs)
    if peak <= 0.99:
        return [wav for wav in wavs]
    scale = 0.99 / peak
    return [wav * scale for wav in wavs]


def save_example_audio(
    output_root: Path,
    split_name: str,
    mixture_id: str,
    mix_audio: torch.Tensor,
    src_1: torch.Tensor,
    src_2: torch.Tensor,
) -> tuple[Path, Path, Path]:
    split_root = output_root / split_name
    mix_dir = split_root / "mix_noisy"
    s1_dir = split_root / "s1"
    s2_dir = split_root / "s2"
    mix_dir.mkdir(parents=True, exist_ok=True)
    s1_dir.mkdir(parents=True, exist_ok=True)
    s2_dir.mkdir(parents=True, exist_ok=True)

    mix_path = mix_dir / f"{mixture_id}.wav"
    s1_path = s1_dir / f"{mixture_id}.wav"
    s2_path = s2_dir / f"{mixture_id}.wav"

    torchaudio.save(str(mix_path), mix_audio.unsqueeze(0), RATE)
    torchaudio.save(str(s1_path), src_1.unsqueeze(0), RATE)
    torchaudio.save(str(s2_path), src_2.unsqueeze(0), RATE)
    return mix_path, s1_path, s2_path


def build_example(
    pair: dict[str, object],
    speakers: dict[str, dict[str, object]],
    noise_bins: dict[str, list[str]],
    output_root: Path,
    split_name: str,
    mix_index: int,
    args: argparse.Namespace,
    rng: random.Random,
) -> dict[str, object]:
    target_len = int(args.segment_sec * RATE)
    speaker_1_id = pair["speaker_1_ID"]
    speaker_2_id = pair["speaker_2_ID"]

    utt_1 = Path(rng.choice(speakers[speaker_1_id]["utterance_paths"]))
    utt_2 = Path(rng.choice(speakers[speaker_2_id]["utterance_paths"]))
    seg_1 = rms_normalize(random_crop_or_pad(load_audio(utt_1), target_len, rng), args.speaker_gain_db)
    seg_2 = rms_normalize(random_crop_or_pad(load_audio(utt_2), target_len, rng), args.speaker_gain_db)

    overlap = rng.uniform(args.min_overlap, args.max_overlap)
    shift = int(round(target_len * (1.0 - overlap)))
    final_len = target_len + shift

    src_1 = torch.zeros(final_len, dtype=torch.float32)
    src_2 = torch.zeros(final_len, dtype=torch.float32)
    src_1[:target_len] = seg_1
    src_2[shift : shift + target_len] = seg_2
    clean_mix = src_1 + src_2

    noise_path = choose_noise_file(noise_bins, final_len / RATE, rng)
    noise = load_audio(Path(noise_path))
    snr_db = rng.uniform(args.snr_min_db, args.snr_max_db)
    noisy_mix = mix_noise_with_snr(clean_mix, noise, snr_db)
    noisy_mix, src_1, src_2 = peak_normalize(noisy_mix, src_1, src_2)

    mixture_id = f"{speaker_1_id}_{speaker_2_id}_{mix_index:07d}"
    mix_path, s1_path, s2_path = save_example_audio(
        output_root=output_root,
        split_name=split_name,
        mixture_id=mixture_id,
        mix_audio=noisy_mix,
        src_1=src_1,
        src_2=src_2,
    )

    return {
        "mixture_ID": mixture_id,
        "mixture_path": str(mix_path),
        "source_1_path": str(s1_path),
        "source_2_path": str(s2_path),
        "speaker_1_ID": speaker_1_id,
        "speaker_2_ID": speaker_2_id,
        "length": final_len,
        "pair_cosine": pair["pair_cosine"],
        "overlap_ratio": overlap,
        "snr_db": snr_db,
        "noise_path": noise_path,
        "utt_1_path": str(utt_1),
        "utt_2_path": str(utt_2),
        "speaker_1_subset": speakers[speaker_1_id]["subset"],
        "speaker_2_subset": speakers[speaker_2_id]["subset"],
        "pair_difficulty": pair.get("pair_difficulty", "hard"),
        "example_type": "two_speaker",
        "num_active_speakers": 2,
        "silence_slot": -1,
    }


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_mapping(speaker_ids: list[str], path: Path) -> None:
    mapping = {speaker_id: idx for idx, speaker_id in enumerate(sorted(speaker_ids))}
    with path.open("w") as handle:
        json.dump(mapping, handle, indent=2)


def generate_split(
    split_name: str,
    pairs: list[dict[str, object]],
    speakers: dict[str, dict[str, object]],
    noise_json_path: Path,
    output_root: Path,
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[list[dict[str, object]], Path | None]:
    if not pairs:
        return [], None

    with noise_json_path.open() as handle:
        noise_bins = json.load(handle)

    rows = []
    if pairs:
        iterator = tqdm(
            enumerate(pairs),
            total=len(pairs),
            desc=f"Creating {split_name} two-speaker mixes",
        )
        for pair_idx, pair in iterator:
            for repeat_idx in range(args.examples_per_pair):
                mix_index = pair_idx * args.examples_per_pair + repeat_idx
                rows.append(
                    build_example(
                        pair=pair,
                        speakers=speakers,
                        noise_bins=noise_bins,
                        output_root=output_root,
                        split_name=split_name,
                        mix_index=mix_index,
                        args=args,
                        rng=rng,
                    )
                )
    rng.shuffle(rows)

    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metadata_dir / f"mixture_{split_name}.csv"
    write_csv(rows, csv_path)
    return rows, csv_path


def write_pair_manifest(path: Path, pairs: list[dict[str, object]]) -> None:
    with path.open("w") as handle:
        json.dump(pairs, handle, indent=2)


def main() -> None:
    args = parse_args()
    args.device = choose_device(args.device)
    if not 0.0 < args.min_overlap <= args.max_overlap <= 1.0:
        raise ValueError("Expected 0 < min_overlap <= max_overlap <= 1.")

    output_root = Path(args.output_root)
    if output_root.exists() and args.force:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    if args.reuse_centroid_pt:
        centroid_pt = Path(args.reuse_centroid_pt)
        speakers, centroids = load_centroid_metadata(centroid_pt)
        centroid_csv = centroid_pt.with_suffix(".csv")
    else:
        speakers_meta = parse_speakers_txt(Path(args.speakers_txt))
        speakers = discover_speakers(
            librispeech_root=Path(args.librispeech_root),
            speakers_meta=speakers_meta,
            min_speaker_utts=args.min_speaker_utts,
            subset_filter=set(args.centroid_subsets) if args.centroid_subsets else None,
        )
        if not speakers:
            raise RuntimeError("No LibriSpeech speakers were discovered with the current filters.")

        teacher_model = load_teacher_model(Path(args.teacher_ckpt), emb_dim=args.emb_dim, device=args.device)
        centroids = compute_speaker_centroids(teacher_model, speakers, device=args.device)
        centroid_pt, centroid_csv = write_centroid_metadata(output_root, speakers, centroids)

    train_pairs = select_hard_easy_pairs(
        speakers=speakers,
        centroids=centroids,
        subsets=args.train_subsets,
        pairs_per_speaker=args.pairs_per_speaker,
        max_pairs=args.max_train_pairs,
        min_similarity=args.min_centroid_sim,
        hard_ratio=args.hard_pair_ratio,
        easy_ratio=args.easy_pair_ratio,
    )
    val_pairs = select_hard_easy_pairs(
        speakers=speakers,
        centroids=centroids,
        subsets=args.val_subsets,
        pairs_per_speaker=max(1, min(args.pairs_per_speaker, 3)),
        max_pairs=args.max_val_pairs,
        min_similarity=args.min_centroid_sim,
        hard_ratio=args.hard_pair_ratio,
        easy_ratio=args.easy_pair_ratio,
    )

    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    write_pair_manifest(metadata_dir / "train_pairs.json", train_pairs)
    write_pair_manifest(metadata_dir / "val_pairs.json", val_pairs)

    train_rows, train_csv = generate_split(
        split_name="train",
        pairs=train_pairs,
        speakers=speakers,
        noise_json_path=Path(args.train_noise_json),
        output_root=output_root,
        args=args,
        rng=rng,
    )
    val_rows, val_csv = generate_split(
        split_name="val",
        pairs=val_pairs,
        speakers=speakers,
        noise_json_path=Path(args.val_noise_json),
        output_root=output_root,
        args=args,
        rng=rng,
    )

    train_speakers = sorted(
        {str(row["speaker_1_ID"]) for row in train_rows} | {str(row["speaker_2_ID"]) for row in train_rows}
    )
    all_speakers = sorted(speakers)
    train_map_path = metadata_dir / "train_mapping.json"
    all_map_path = metadata_dir / "all_speakers_mapping.json"
    write_mapping(train_speakers, train_map_path)
    write_mapping(all_speakers, all_map_path)

    manifest = {
        "librispeech_root": args.librispeech_root,
        "speakers_txt": args.speakers_txt,
        "teacher_ckpt": args.teacher_ckpt,
        "device": args.device,
        "output_root": args.output_root,
        "segment_sec": args.segment_sec,
        "min_overlap": args.min_overlap,
        "max_overlap": args.max_overlap,
        "pairs_per_speaker": args.pairs_per_speaker,
        "examples_per_pair": args.examples_per_pair,
        "hard_pair_ratio": args.hard_pair_ratio,
        "easy_pair_ratio": args.easy_pair_ratio,
        "min_speaker_utts": args.min_speaker_utts,
        "train_subsets": args.train_subsets,
        "val_subsets": args.val_subsets,
        "num_discovered_speakers": len(speakers),
        "num_centroids": len(centroids),
        "num_train_pairs": len(train_pairs),
        "num_val_pairs": len(val_pairs),
        "num_train_examples": len(train_rows),
        "num_val_examples": len(val_rows),
        "num_train_hard_pairs": sum(row.get("pair_difficulty") == "hard" for row in train_pairs),
        "num_train_easy_pairs": sum(row.get("pair_difficulty") == "easy" for row in train_pairs),
        "num_val_hard_pairs": sum(row.get("pair_difficulty") == "hard" for row in val_pairs),
        "num_val_easy_pairs": sum(row.get("pair_difficulty") == "easy" for row in val_pairs),
        "centroid_pt": str(centroid_pt),
        "centroid_csv": str(centroid_csv),
        "train_csv": str(train_csv) if train_csv else None,
        "val_csv": str(val_csv) if val_csv else None,
        "train_mapping_json": str(train_map_path),
        "all_speakers_mapping_json": str(all_map_path),
    }
    with (output_root / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
