import argparse
import os
import random
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from tqdm import tqdm

sys.path.append("/home/sidcs/codebase/wavlm_dual_embedding")
from model import SpeakerEncoderDualWrapper

sys.path.append("/home/sidcs/codebase")
from wavlm_single_embedding.model import SpeakerEncoderWrapper as SingleSpkEncoder


def mix_with_snr(clean, noise, snr_db):
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


def sample_snr(rng):
    r = rng.random()
    if r < 0.4:
        return rng.uniform(-5, 5)
    if r < 0.8:
        return rng.uniform(5, 15)
    return rng.uniform(15, 25)


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


def load_dual_model(ckpt_path, emb_dim=256, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = strip_dual_model_weights(ckpt["state_dict"])
    model = SpeakerEncoderDualWrapper(emb_dim=emb_dim)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def load_teacher_model(teacher_ckpt, device="cuda"):
    teacher = SingleSpkEncoder(emb_dim=256).to(device)
    ckpt = torch.load(teacher_ckpt, map_location=device)
    state = ckpt["state_dict"]

    filtered = {}
    for k, v in state.items():
        if not k.startswith("model."):
            continue
        if "arcface" in k or "arc_face" in k:
            continue
        filtered[k.replace("model.", "", 1)] = v

    teacher.load_state_dict(filtered, strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def load_audio(path):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != 16000:
        raise ValueError(f"Expected 16 kHz audio, got {sr} for {path}")
    return torch.from_numpy(wav)


def get_teacher_emb(teacher_model, wav_path, device):
    wav = load_audio(wav_path).to(device).unsqueeze(0)
    with torch.no_grad():
        emb = teacher_model(wav)
    return emb.squeeze(0)


def cosine(a, b):
    return float((a @ b) / (a.norm() * b.norm() + 1e-8))


def parse_speaker_ids(filename):
    stem = Path(filename).stem
    utts = stem.split("_")
    speaker_ids = [utt.split("-")[0] for utt in utts]
    if len(speaker_ids) != 3:
        raise ValueError(f"Expected 3 speaker IDs in {filename}, got {speaker_ids}")
    return speaker_ids


def best_two_of_three_assignment(sim_matrix):
    best = None
    best_sum = None
    second_best_sum = None

    for pred_perm in permutations(range(3), 2):
        score = sim_matrix[0, pred_perm[0]] + sim_matrix[1, pred_perm[1]]
        if best_sum is None or score > best_sum:
            second_best_sum = best_sum
            best_sum = score
            best = pred_perm
        elif second_best_sum is None or score > second_best_sum:
            second_best_sum = score

    return best, float(best_sum), float(second_best_sum if second_best_sum is not None else best_sum)


def compute_clustering_metrics(embs, labels):
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
    e = embs / norms
    n = e.shape[0]

    same, diff = [], []
    for i in range(n):
        for j in range(i + 1, n):
            cos = float(np.dot(e[i], e[j]))
            if labels[i] == labels[j]:
                same.append(cos)
            else:
                diff.append(cos)

    same_mean = np.mean(same) if same else 0.0
    diff_mean = np.mean(diff) if diff else 0.0
    separation = same_mean - diff_mean

    speakers = np.unique(labels)
    k = len(speakers)
    pred = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(e)

    cluster_acc = cluster_accuracy(pred, labels)
    nmi = normalized_mutual_info_score(labels, pred)
    ari = adjusted_rand_score(labels, pred)

    try:
        silhouette = silhouette_score(e, labels)
    except Exception:
        silhouette = float("nan")

    return {
        "same_mean_cos": float(same_mean),
        "diff_mean_cos": float(diff_mean),
        "separation": float(separation),
        "cluster_acc": float(cluster_acc),
        "nmi": float(nmi),
        "ari": float(ari),
        "silhouette": float(silhouette),
    }


def cluster_accuracy(pred_labels, true_labels):
    total = 0
    pred = np.array(pred_labels)
    true = np.array(true_labels)
    for c in np.unique(pred):
        idx = pred == c
        true_subset = true[idx]
        if len(true_subset) == 0:
            continue
        total += pd.Series(true_subset).value_counts().iloc[0]
    return total / len(true)


def evaluate_one_file(filename, root, dual_model, teacher_model, noise_files, rng, device, add_noise):
    mix_path = root / "mix_clean" / filename
    src_paths = [root / "s1" / filename, root / "s2" / filename, root / "s3" / filename]
    speaker_ids = parse_speaker_ids(filename)

    mix = load_audio(mix_path)
    noise_path = None
    snr_db = None
    if add_noise and noise_files:
        noise_path = rng.choice(noise_files)
        noise_wav = load_audio(noise_path)
        snr_db = sample_snr(rng)
        mix = mix_with_snr(mix, noise_wav, snr_db)

    mix = mix.to(device).unsqueeze(0)
    with torch.no_grad():
        pred = dual_model(mix).squeeze(0)

    teacher_embs = [get_teacher_emb(teacher_model, str(path), device) for path in src_paths]
    sim_matrix = np.asarray(
        [[cosine(pred[pred_idx], teacher_embs[src_idx]) for src_idx in range(3)] for pred_idx in range(2)],
        dtype=np.float32,
    )

    top1_idx = [int(np.argmax(sim_matrix[row])) for row in range(2)]
    top1_cos = [float(sim_matrix[row, top1_idx[row]]) for row in range(2)]
    top1_distinct = top1_idx[0] != top1_idx[1]

    best_pair, best_sum, second_best_sum = best_two_of_three_assignment(sim_matrix)
    chosen_src_idxs = [int(best_pair[0]), int(best_pair[1])]
    chosen_speakers = [speaker_ids[idx] for idx in chosen_src_idxs]
    dropped_src_idx = int(next(idx for idx in range(3) if idx not in chosen_src_idxs))
    dropped_speaker = speaker_ids[dropped_src_idx]

    assigned_cos = [float(sim_matrix[0, chosen_src_idxs[0]]), float(sim_matrix[1, chosen_src_idxs[1]])]

    row = {
        "filename": filename,
        "speaker_1_id": speaker_ids[0],
        "speaker_2_id": speaker_ids[1],
        "speaker_3_id": speaker_ids[2],
        "pred0_top1_speaker": speaker_ids[top1_idx[0]],
        "pred1_top1_speaker": speaker_ids[top1_idx[1]],
        "pred0_top1_cos": top1_cos[0],
        "pred1_top1_cos": top1_cos[1],
        "top1_distinct": int(top1_distinct),
        "best_pair_pred0_speaker": chosen_speakers[0],
        "best_pair_pred1_speaker": chosen_speakers[1],
        "best_pair_pred0_cos": assigned_cos[0],
        "best_pair_pred1_cos": assigned_cos[1],
        "best_pair_sum": best_sum,
        "best_pair_avg": best_sum / 2.0,
        "best_pair_margin": best_sum - second_best_sum,
        "dropped_speaker": dropped_speaker,
        "noise_path": str(noise_path) if noise_path is not None else "",
        "snr_db": float(snr_db) if snr_db is not None else np.nan,
        "sim_p0_s1": float(sim_matrix[0, 0]),
        "sim_p0_s2": float(sim_matrix[0, 1]),
        "sim_p0_s3": float(sim_matrix[0, 2]),
        "sim_p1_s1": float(sim_matrix[1, 0]),
        "sim_p1_s2": float(sim_matrix[1, 1]),
        "sim_p1_s3": float(sim_matrix[1, 2]),
    }
    assigned_embs = [pred[0].detach().cpu().numpy(), pred[1].detach().cpu().numpy()]
    assigned_labels = [chosen_speakers[0], chosen_speakers[1]]
    return row, assigned_embs, assigned_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/home/sidcs/datasets/LibriMix/LibriMix/3sp/Libri3Mix_ovl50to80/wav16k/min/test",
    )
    parser.add_argument(
        "--dual-ckpt",
        type=str,
        default="/home/sidcs/model_ckpts/librispeech_asp_ft_wavlm_linear_dualemb_tr360/best-epoch=49-val_separation=0.000.ckpt",
    )
    parser.add_argument(
        "--teacher-ckpt",
        type=str,
        default="/home/sidcs/model_ckpts/librispeech_asp_wavlm_tr360/best-epoch=62-val_separation=0.000.ckpt",
    )
    parser.add_argument(
        "--noise-dir",
        type=str,
        default="/home/sidcs/datasets/LibriMix/LibriMix/wham_noise/tt",
    )
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--add-noise", action="store_true", default=True)
    parser.add_argument("--no-add-noise", dest="add_noise", action="store_false")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mix_dir = dataset_root / "mix_clean"
    filenames = sorted([name for name in os.listdir(mix_dir) if name.endswith(".wav")])
    if args.limit is not None:
        filenames = filenames[: args.limit]

    noise_files = []
    if args.add_noise and os.path.isdir(args.noise_dir):
        noise_files = sorted(
            [str(Path(args.noise_dir) / name) for name in os.listdir(args.noise_dir) if name.endswith(".wav")]
        )

    rng = random.Random(args.seed)

    print(f"Using device: {device}")
    print(f"Num mixtures: {len(filenames)}")
    print(f"Add noise: {args.add_noise} (noise files: {len(noise_files)})")

    dual_model = load_dual_model(args.dual_ckpt, device=device)
    teacher_model = load_teacher_model(args.teacher_ckpt, device=device)

    rows = []
    all_embs = []
    all_labels = []
    for filename in tqdm(filenames, desc="3sp eval"):
        row, assigned_embs, assigned_labels = evaluate_one_file(
            filename=filename,
            root=dataset_root,
            dual_model=dual_model,
            teacher_model=teacher_model,
            noise_files=noise_files,
            rng=rng,
            device=device,
            add_noise=args.add_noise,
        )
        rows.append(row)
        all_embs.extend(assigned_embs)
        all_labels.extend(assigned_labels)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "3sp_any2_teacher_match.csv"
    df.to_csv(csv_path, index=False)

    summary = {
        "num_mixtures": int(len(df)),
        "num_embeddings": int(len(all_embs)),
        "top1_distinct_rate": float(df["top1_distinct"].mean()),
        "pred0_top1_cos_mean": float(df["pred0_top1_cos"].mean()),
        "pred1_top1_cos_mean": float(df["pred1_top1_cos"].mean()),
        "best_pair_avg_mean": float(df["best_pair_avg"].mean()),
        "best_pair_sum_mean": float(df["best_pair_sum"].mean()),
        "best_pair_margin_mean": float(df["best_pair_margin"].mean()),
        "best_pair_pred0_cos_mean": float(df["best_pair_pred0_cos"].mean()),
        "best_pair_pred1_cos_mean": float(df["best_pair_pred1_cos"].mean()),
    }
    clustering = compute_clustering_metrics(np.asarray(all_embs), np.asarray(all_labels))
    summary.update(clustering)

    dropped_counts = df["dropped_speaker"].value_counts().to_dict()
    print(f"Saved per-mixture results to: {csv_path}")
    print("\nSummary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("\nDropped-speaker counts (which GT speaker was left unmatched by the best 2-of-3 pairing):")
    for key, value in list(dropped_counts.items())[:20]:
        print(f"{key}: {value}")

    pd.DataFrame([summary]).to_csv(out_dir / "3sp_any2_teacher_match_summary.csv", index=False)


if __name__ == "__main__":
    main()
