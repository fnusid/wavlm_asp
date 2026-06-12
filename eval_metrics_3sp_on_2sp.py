import argparse
import itertools
import os
import random
import sys

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from tqdm import tqdm

from model import SpeakerEncoderDualWrapper

sys.path.append("/home/sidcs/codebase")
from wavlm_single_embedding.model import SpeakerEncoderWrapper as SingleSpkEncoder


DEFAULT_META = (
    "/home/sidcs/datasets/LibriMix/LibriMix/Libriuni_05_08/"
    "Libri2Mix_ovl50to80/wav16k/min/metadata/mixture_test_mix_clean.csv"
)
DEFAULT_CKPT = (
    "/home/sidcs/model_ckpts/librispeech_asp_3spft_wavlm_linear_dualemb_tr360/"
    "best-epoch=54-val_separation=0.000.ckpt"
)
DEFAULT_TEACHER_CKPT = (
    "/home/sidcs/model_ckpts/librispeech_asp_wavlm_tr360/"
    "best-epoch=62-val_separation=0.000.ckpt"
)
DEFAULT_NOISE_DIR = "/home/sidcs/datasets/LibriMix/LibriMix/wham_noise/tt"


def load_audio_mono(path):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    return torch.from_numpy(wav), sr


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
    target_np = clean_power / (10 ** (snr_db / 10))
    scale = torch.sqrt(target_np / (noise_power + 1e-8))
    return clean + scale * noise


def strip_dual_model_weights(state):
    new_state = {}
    for key, value in state.items():
        if not key.startswith("model."):
            continue
        key = key.replace("model.", "", 1)
        if key.startswith("single_sp_model.") or key.startswith("arcface_loss."):
            continue
        new_state[key] = value
    return new_state


def load_dual_model(ckpt_path, emb_dim=256, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = strip_dual_model_weights(ckpt["state_dict"])
    model = SpeakerEncoderDualWrapper(emb_dim=emb_dim)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def load_teacher_model(ckpt_path, device="cuda"):
    teacher = SingleSpkEncoder().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"]

    filtered = {}
    for key, value in state.items():
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


def parse_metadata(csv_path, limit=None):
    metadata = []
    with open(csv_path, "r") as handle:
        next(handle)
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) != 7:
                continue
            _, mix_path, src1, src2, spk1, spk2, _ = parts
            metadata.append(
                {
                    "mix_path": mix_path,
                    "src1": src1,
                    "src2": src2,
                    "spk1": int(spk1),
                    "spk2": int(spk2),
                }
            )
            if limit is not None and len(metadata) >= limit:
                break
    return metadata


def get_teacher_emb(teacher_model, wav_path, device="cuda"):
    wav, _ = load_audio_mono(wav_path)
    wav = wav.to(device).unsqueeze(0)
    with torch.no_grad():
        emb = teacher_model(wav)
    return emb.squeeze(0)


def cosine(a, b):
    return (a @ b) / (a.norm() * b.norm() + 1e-8)


def extract_embeddings_with_teacher(
    dual_model,
    teacher_model,
    metadata,
    noise_files,
    add_noise=True,
    device="cuda",
    seed=44,
):
    rng = random.Random(seed)
    all_embs = []
    all_labels = []
    unused_slot_counts = [0, 0, 0]

    iterator = tqdm(metadata, desc="Evaluating 3sp model on 2sp mixtures")
    for entry in iterator:
        mix, _ = load_audio_mono(entry["mix_path"])

        if add_noise and noise_files:
            noise_path = rng.choice(noise_files)
            noise_wav, _ = load_audio_mono(noise_path)
            r = rng.random()
            if r < 0.4:
                snr = rng.uniform(-5, 5)
            elif r < 0.8:
                snr = rng.uniform(5, 15)
            else:
                snr = rng.uniform(15, 25)
            mix = mix_with_snr(mix, noise_wav, snr)

        mix = mix.to(device).unsqueeze(0)
        with torch.no_grad():
            pred = dual_model(mix).squeeze(0)

        t1 = get_teacher_emb(teacher_model, entry["src1"], device)
        t2 = get_teacher_emb(teacher_model, entry["src2"], device)
        targets = [t1, t2]
        speakers = [entry["spk1"], entry["spk2"]]

        best_score = float("-inf")
        best_choice = None
        for pred_pair in itertools.permutations(range(3), 2):
            score = cosine(pred[pred_pair[0]], targets[0]) + cosine(pred[pred_pair[1]], targets[1])
            if score > best_score:
                best_score = score
                best_choice = pred_pair

            swapped_score = cosine(pred[pred_pair[0]], targets[1]) + cosine(pred[pred_pair[1]], targets[0])
            if swapped_score > best_score:
                best_score = swapped_score
                best_choice = (pred_pair[0], pred_pair[1], "swap")

        if len(best_choice) == 2:
            mapped = [
                (pred[best_choice[0]], speakers[0]),
                (pred[best_choice[1]], speakers[1]),
            ]
            unused = ({0, 1, 2} - set(best_choice)).pop()
        else:
            mapped = [
                (pred[best_choice[0]], speakers[1]),
                (pred[best_choice[1]], speakers[0]),
            ]
            unused = ({0, 1, 2} - {best_choice[0], best_choice[1]}).pop()

        unused_slot_counts[unused] += 1
        for emb, label in mapped:
            all_embs.append(emb.cpu().numpy())
            all_labels.append(label)

    return np.vstack(all_embs), np.array(all_labels), unused_slot_counts


def cluster_accuracy(pred_labels, true_labels):
    from collections import Counter

    pred = np.array(pred_labels)
    true = np.array(true_labels)
    total = 0
    for cluster_id in np.unique(pred):
        idx = pred == cluster_id
        true_subset = true[idx]
        if len(true_subset) == 0:
            continue
        total += Counter(true_subset).most_common(1)[0][1]
    return total / len(true)


def compute_clustering_metrics(embs, labels):
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
    e = embs / norms
    n = e.shape[0]

    same = []
    diff = []
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
    kmeans = KMeans(n_clusters=len(speakers), n_init=10, random_state=0)
    pred = kmeans.fit_predict(e)

    try:
        silhouette = silhouette_score(e, labels)
    except Exception:
        silhouette = float("nan")

    return {
        "same_mean_cos": float(same_mean),
        "diff_mean_cos": float(diff_mean),
        "separation": float(separation),
        "cluster_acc": float(cluster_accuracy(pred, labels)),
        "nmi": float(normalized_mutual_info_score(labels, pred)),
        "ari": float(adjusted_rand_score(labels, pred)),
        "silhouette": float(silhouette),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", default=DEFAULT_META)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--teacher-ckpt", default=DEFAULT_TEACHER_CKPT)
    parser.add_argument("--noise-dir", default=DEFAULT_NOISE_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--no-noise", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    metadata = parse_metadata(args.meta, limit=args.limit)
    print(f"Loaded {len(metadata)} 2sp mixtures from {args.meta}")

    noise_files = []
    if not args.no_noise and os.path.isdir(args.noise_dir):
        noise_files = [
            os.path.join(args.noise_dir, name)
            for name in os.listdir(args.noise_dir)
            if name.endswith(".wav")
        ]
    print(f"Noise files available: {len(noise_files)}")

    teacher = load_teacher_model(args.teacher_ckpt, device=device)
    dual = load_dual_model(args.ckpt, device=device)

    embs, labels, unused_slot_counts = extract_embeddings_with_teacher(
        dual_model=dual,
        teacher_model=teacher,
        metadata=metadata,
        noise_files=noise_files,
        add_noise=not args.no_noise,
        device=device,
        seed=args.seed,
    )
    print(f"Extracted {len(embs)} embeddings across {len(np.unique(labels))} speakers")
    print(f"Unused predicted slot counts: {unused_slot_counts}")

    res = compute_clustering_metrics(embs, labels)
    print("\n=== Clustering / Separation Metrics ===")
    for key, value in res.items():
        print(f"{key} = {value:.6f}")


if __name__ == "__main__":
    main()
