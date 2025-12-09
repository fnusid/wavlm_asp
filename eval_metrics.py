import os
import csv
import torch
import torchaudio
import numpy as np

from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
    roc_curve,
)

from model import SpeakerEncoderWrapper  # your single-speaker model
import random
add_noise = True  # set True to enable noise corruption
noise_dir = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/wham_noise/tt"
noise_files = [
    os.path.join(noise_dir, f)
    for f in os.listdir(noise_dir)
    if f.endswith(".wav")
]

random.seed(44)
def mix_with_snr(clean, noise, snr_db):
    """
    clean, noise: torch tensors [T]
    snr_db: desired SNR in dB
    """
    if noise.ndim > 1:
        noise = noise.mean(0)
    noise = noise.to("cuda")
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


# -------------------------------------------------
# 1. Load model
# -------------------------------------------------
def load_model(ckpt_path, emb_dim=256, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)

    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state[k.replace("model.", "", 1)] = v
        else:
            new_state[k] = v

    model = SpeakerEncoderWrapper(emb_dim=emb_dim)
    model.load_state_dict(new_state, strict=False)
    model.to(device)
    model.eval()
    return model


# -------------------------------------------------
# 2. Embedding cache + helpers
# -------------------------------------------------
def get_speaker_id_from_path(wav_path):
    """
    Assumes LibriSpeech-style path: .../<spk_id>/<chapter>/<file>.flac
    """
    return int(wav_path.split("/")[-3])


def embed_utterance(model, wav_path, cache, device):
    """
    Compute embedding for a single wav_path; use cache if already done.
    cache: dict[path -> np.array(D)]
    """
    if wav_path in cache:
        return cache[wav_path]

    wav, sr = torchaudio.load(wav_path)
    wav = wav.mean(dim=0)               # mono
    wav = wav.to(device).unsqueeze(0)   # [1, T]
    if add_noise and noise_files:
        noise_p = random.choice(noise_files)
        noise_wav, _ = torchaudio.load(noise_p)

        r = random.random()
        if r < 0.4:
            snr = random.uniform(-5, 5)
        elif r < 0.8:
            snr = random.uniform(5, 15)
        else:
            snr = random.uniform(15, 25)

        wav = mix_with_snr(wav, noise_wav.squeeze(0), snr)

    with torch.no_grad():
        emb = model(wav)                # [1, D]

    emb_np = emb.squeeze(0).cpu().numpy()
    cache[wav_path] = emb_np
    return emb_np


# -------------------------------------------------
# 3. Full dev-clean embeddings for clustering/separation
# -------------------------------------------------
def extract_all_embeddings_from_list(model, path_list, device="cuda"):
    """
    path_list: list of wav/flac paths
    Returns:
        embs   [N, D]
        labels [N]  (speaker IDs)
    """
    cache = {}
    all_embs = []
    all_labels = []

    for i, p in enumerate(path_list):
        e = embed_utterance(model, p, cache, device)
        all_embs.append(e)
        all_labels.append(get_speaker_id_from_path(p))

        if (i + 1) % 100 == 0:
            print(f"[extract_all] {i + 1}/{len(path_list)} done")

    embs = np.vstack(all_embs)
    labels = np.array(all_labels)
    return embs, labels, cache


# -------------------------------------------------
# 4. Separation
# -------------------------------------------------
def compute_separation(embs, labels):
    """
    embs:   [N, D]
    labels: [N]
    Returns:
        separation, same_mean, diff_mean
    """
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
    e = embs / norms

    N = len(labels)
    same, diff = [], []

    for i in range(N):
        for j in range(i + 1, N):
            cos = float(np.dot(e[i], e[j]))
            if labels[i] == labels[j]:
                same.append(cos)
            else:
                diff.append(cos)

    same_mean = float(np.mean(same)) if len(same) > 0 else 0.0
    diff_mean = float(np.mean(diff)) if len(diff) > 0 else 0.0
    separation = same_mean - diff_mean

    return separation, same_mean, diff_mean


# -------------------------------------------------
# 5. Clustering metrics
# -------------------------------------------------
def cluster_accuracy(pred_labels, true_labels):
    pred = np.array(pred_labels)
    true = np.array(true_labels)
    clusters = np.unique(pred)

    total = 0
    for c in clusters:
        mask = (pred == c)
        true_subset = true[mask]
        if len(true_subset) == 0:
            continue
        most_common = Counter(true_subset).most_common(1)[0][1]
        total += most_common

    return total / len(true)


def compute_clustering_metrics(embs, labels, n_clusters=None):
    if n_clusters is None:
        n_clusters = len(np.unique(labels))

    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
    e = embs / norms

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    pred = kmeans.fit_predict(e)

    acc = cluster_accuracy(pred, labels)
    nmi = normalized_mutual_info_score(labels, pred)
    ari = adjusted_rand_score(labels, pred)

    try:
        sil = silhouette_score(e, labels)
    except Exception:
        sil = float("nan")

    return {
        "cluster_acc": acc,
        "nmi": nmi,
        "ari": ari,
        "silhouette": sil,
    }


# -------------------------------------------------
# 6. EER from your pairs CSV
# -------------------------------------------------
def load_trials(csv_path):
    """
    CSV format:
    utt1,utt2,label
    /path/to/utt1.flac,/path/to/utt2.flac,0/1
    """
    paths1, paths2, labels = [], [], []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if len(row) < 3:
                continue
            # skip header if present
            if i == 0 and row[0].lower() in ["utt1", "path1", "u1"]:
                continue
            p1, p2, lab = row[0].strip(), row[1].strip(), row[2].strip()
            paths1.append(p1)
            paths2.append(p2)
            labels.append(int(lab))
    return paths1, paths2, np.array(labels)


def load_trials_vox1_txt(txt_path):
    """
    TXT format (same as CSV):
    label,utt1,utt2

    Example line:
    1,id01234/xxxxxx.wav,id05678/yyyyyy.wav
    """
  
    path_prefix = "/mnt/disks/data/datasets/Datasets/voxceleb/vox1/eval/wav"
    paths1, paths2, labels = [], [], []

    with open(txt_path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            parts = line.split(" ")
            if len(parts) < 3:
                continue

            # header check
            if i == 0 and parts[0].lower() in ["utt1", "path1", "u1", "label"]:
                continue

            lab = int(parts[0].strip())
            p1  = parts[1].strip()
            p2  = parts[2].strip()

            paths1.append(f"{path_prefix}/{p1}")
            paths2.append(f"{path_prefix}/{p2}")
            labels.append(lab)

    return paths1, paths2, np.array(labels)


def compute_eer_from_trials(model, trial_paths1, trial_paths2, trial_labels, cache, device="cuda"):
    """
    model: embedding model
    trial_paths1, trial_paths2: lists of paths (same length)
    trial_labels: [N_pairs] np.array (0/1)
    cache: dict[path -> embedding] reused from global embedding extraction
    """
    scores = []

    for i, (p1, p2) in enumerate(zip(trial_paths1, trial_paths2)):
        e1 = embed_utterance(model, p1, cache, device)
        e2 = embed_utterance(model, p2, cache, device)

        # cosine similarity
        e1 = e1 / (np.linalg.norm(e1) + 1e-10)
        e2 = e2 / (np.linalg.norm(e2) + 1e-10)
        cos = float(np.dot(e1, e2))
        scores.append(cos)

        if (i + 1) % 1000 == 0:
            print(f"[trials] {i + 1}/{len(trial_labels)} pairs done")

    scores = np.array(scores)

    fpr, tpr, _ = roc_curve(trial_labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer


# -------------------------------------------------
# 7. Main
# -------------------------------------------------
if __name__ == "__main__":
    # ---- EDIT THESE PATHS ----
    METADATA_TXT = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/LibriSpeech/libri_test_clean.txt"
    CKPT = "/mnt/disks/data/model_ckpts/librispeech_asp_wavlm_tr360/best-epoch=62-val_separation=0.000.ckpt"
    TRIALS_CSV = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/LibriSpeech/dev_clean_sp_ver_pairs.csv"  # <-- your CSV with utt1,utt2,label
    TRIALS_CSV_VOX1 = "/mnt/disks/data/datasets/Datasets/Vox1_sp_ver/svs.txt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device='cpu'

    # 1) Load model
    model = load_model(CKPT, emb_dim=256, device=device)

    # 2) Load metadata list (full dev-clean)
    with open(METADATA_TXT, "r") as f:
        metadata_paths = [x.strip() for x in f if x.strip()]
    print(f"Total dev-clean files: {len(metadata_paths)}")

    # 3) Extract ALL embeddings + cache
    embs, labels, cache = extract_all_embeddings_from_list(model, metadata_paths, device=device)
    print(f"Embeddings shape: {embs.shape}, unique speakers: {len(np.unique(labels))}")

    # 4) Separation
    sep, same_mean, diff_mean = compute_separation(embs, labels)
    print("\n=== Separation Metrics (dev-clean) ===")
    print(f"same_mean_cos = {same_mean:.4f}")
    print(f"diff_mean_cos = {diff_mean:.4f}")
    print(f"separation    = {sep:.4f}")

    # 5) Clustering
    clust = compute_clustering_metrics(embs, labels)
    print("\n=== Clustering Metrics (dev-clean) ===")
    print(f"cluster_acc = {clust['cluster_acc']:.4f}")
    print(f"nmi         = {clust['nmi']:.4f}")
    print(f"ari         = {clust['ari']:.4f}")
    print(f"silhouette  = {clust['silhouette']:.4f}")

    # # 6) EER from your trials CSV
    # paths1, paths2, labs = load_trials_vox1_txt(TRIALS_CSV_VOX1)
    # cache={}
    # print(f"\nTotal trials loaded: {len(labs)}")

    # eer = compute_eer_from_trials(model, paths1, paths2, labs, cache, device=device)
    # print("\n=== Verification (dev-clean trials) ===")
    # print(f"EER = {eer * 100:.2f}%")