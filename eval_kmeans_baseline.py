import os
import sys
import random
import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from transformers import WavLMModel, WavLMConfig

# -----------------------------
# Global noise config
# -----------------------------
add_noise = True  # set True to enable noise corruption
noise_dir = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/wham_noise/tt"
noise_files = [
    os.path.join(noise_dir, f)
    for f in os.listdir(noise_dir)
    if f.endswith(".wav")
]

random.seed(44)


# =====================================================================
# 0) Utility: mix with SNR  (same as your dual script)
# =====================================================================
def mix_with_snr(clean, noise, snr_db):
    """
    clean, noise: torch tensors [T]
    snr_db: desired SNR in dB
    """
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


# =====================================================================
# 1) LibriMix metadata parser  (same as your dual script)
# =====================================================================
def parse_metadata(csv_path):
    metadata = []
    filename = os.path.basename(csv_path)
    '''
    mixture_ID,mixture_path,source_1_path,source_2_path,source_3_path,speaker_1_ID,speaker_2_ID,speaker_3_ID,length
    '''
    with open(csv_path, "r") as f:
        header = next(f)
        for line in f:
            parts = line.strip().split(",")

            if filename.split("_")[-1] == "both.csv":
                if len(parts) != 10:
                    continue
                mix_id, mix_path, src1, src2, src3, spk1, spk2, spk3, noise, length = parts
            else:
          
                if len(parts) != 9:
                    continue
                mix_id, mix_path, src1, src2, src3, spk1, spk2, spk3, length = parts

            if spk1 == "speaker_1_ID":
                continue

            metadata.append({
                "mix_path": mix_path,
                "src1": src1,
                "src2": src2,
                "src3": src3,
                "spk1": int(spk1),
                "spk2": int(spk2),
                "spk3": int(spk3)
            })

    return metadata


# =====================================================================
# 2) WavLM backbone (same as in your dual model, but used directly)
# =====================================================================
def load_wavlm(device="cuda"):
    config = WavLMConfig.from_pretrained("microsoft/wavlm-base-plus")
    model = WavLMModel.from_pretrained(
        "microsoft/wavlm-base-plus",
        config=config,
        ignore_mismatched_sizes=True
    )
    model.to(device).eval()
    model.requires_grad_(False)
    return model


def get_wavlm_features(wavlm, wav, sr, device="cuda"):
    """
    wav: 1D torch tensor [T] (mixture, mono)
    sr : sampling rate
    Returns:
        feats: np.ndarray [T_frames, D]
    """
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000

    wav = wav.to(device).unsqueeze(0)  # [1, T]
    with torch.no_grad():
        out = wavlm(wav)
        feats = out.last_hidden_state[0]  # [T_frames, D]
    return feats.cpu().numpy()  # [T_frames, D]


# =====================================================================
# 3) WavLM + KMeans embedding extraction (2 clusters → 2 embeddings)
# =====================================================================
def extract_kmeans_embeddings_wavlm(
    wavlm,
    metadata,
    device="cuda",
    verbose=True,
):
    """
    For each mixture:
      - Load mixture (+ optional WHAM noise).
      - Pass mixture through WavLM to get frame-level features [T_frames, D].
      - Run KMeans with K=2 on frames.
      - For each cluster, average all frames in that cluster → cluster embedding.
      - Assign each cluster a speaker ID via majority vote using frame-wise
        dominant speaker labels from clean src1/src2 (only used for labeling).
    Returns:
      all_embs   -> [N, D]
      all_labels -> [N]
    """
    all_embs = []
    all_labels = []

    iterator = tqdm(metadata, desc="Extracting WavLM+KMeans embeddings", disable=not verbose)

    for entry in iterator:
        mix_path = entry["mix_path"]
        src1_path = entry["src1"]
        src2_path = entry["src2"]
        src3_path = entry["src3"]
        spk1_id = entry["spk1"]
        spk2_id = entry["spk2"]
        spk3_id = entry["spk3"]

        # ------------ Load mixture ------------
        mix, sr = torchaudio.load(mix_path)
        mix = mix.mean(0)  # mono

        # ------------ Add WHAM noise (optional) ------------
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

            mix = mix_with_snr(mix, noise_wav.squeeze(0), snr)

        # ------------ Load clean sources (for cluster->speaker labeling only) ------------
        src1, sr1 = torchaudio.load(src1_path)
        src2, sr2 = torchaudio.load(src2_path)
        src3, sr3 = torchaudio.load(src3_path)
        src1 = src1.mean(0)
        src2 = src2.mean(0)
        src3 = src3.mean(0)

        if sr1 != 16000:
            src1 = torchaudio.functional.resample(src1, sr1, 16000)
        if sr2 != 16000:
            src2 = torchaudio.functional.resample(src2, sr2, 16000)
        if sr3 != 16000:
            src3 = torchaudio.functional.resample(src3, sr3, 16000)

        # ------------ WavLM features ------------
        feats = get_wavlm_features(wavlm, mix, sr, device=device)  # [T_frames, D]
        T_frames, D = feats.shape
        if T_frames < 2:
            continue

        # approximate hop in samples for mapping frames to sources
        hop_samples = len(src1) / float(T_frames)

        # ------------ frame-wise dominant speaker labels (for voting only) ------------
        src1_np = src1.cpu().numpy()
        src2_np = src2.cpu().numpy()
        src3_np = src3.cpu().numpy()

        frame_dom = []
        T_sig = min(len(src1_np), len(src2_np), len(src3_np))
        for i in range(T_frames):
            start = int(i * hop_samples)
            end = int((i + 1) * hop_samples)
            if start >= T_sig:
                frame_dom.append(0)
                continue
            end = min(end, T_sig)
            seg1 = src1_np[start:end]
            seg2 = src2_np[start:end]
            seg3 = src3_np[start:end]
            e1 = np.mean(seg1**2) + 1e-10
            e2 = np.mean(seg2**2) + 1e-10
            e3 = np.mean(seg3**2) + 1e-10
            # 1 = speaker1 dominant, 2 = speaker2 dominant
            if e1>e2:
                if e1>e3:
                    frame_dom.append(1)
                else:
                    frame_dom.append(3)
            else:
                if e2 > e3:
                    frame_dom.append(2)
                else:
                    frame_dom.append(3)
            # frame_dom.append(1 if e1 >= e2 else 2)
        frame_dom = np.array(frame_dom, dtype=np.int64)

        # ------------ KMeans on WavLM features (K=2 speakers) ------------
        kmeans = KMeans(n_clusters=3, n_init=10, random_state=0)
        cluster_ids = kmeans.fit_predict(feats)  # [T_frames]

        # ------------ For each cluster: average features + majority-vote speaker ------------
        for c in range(3):
            mask = cluster_ids == c
            if not np.any(mask):
                continue

            cluster_feats = feats[mask]          # [Nc, D]
            cluster_emb = cluster_feats.mean(axis=0)  # [D]

            cluster_frame_labels = frame_dom[mask]
            # count how many frames mapped to spk1 vs spk2
            count1 = np.sum(cluster_frame_labels == 1)
            count2 = np.sum(cluster_frame_labels == 2)
            count3 = np.sum(cluster_frame_labels == 3)

            if count1 == 0 and count2 == 0 and count3 == 0:
                # no usable frames, skip
                continue

            if count1 >= count2:
                if count1>=count3:
                    spk_id = spk1_id
                else:
                    spk_id = spk3_id
            else:
                if count2 >=count3:
                    spk_id = spk2_id
                else:
                    spk_id = spk3_id


            all_embs.append(cluster_emb)
            all_labels.append(spk_id)

    if len(all_embs) == 0:
        return np.zeros((0, 768)), np.zeros((0,))
    return np.vstack(all_embs), np.array(all_labels)


# =====================================================================
# 4) Clustering + Separation Metrics (same as your script)
# =====================================================================
def compute_clustering_metrics(embs, labels):
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
    e = embs / norms
    N = e.shape[0]

    same, diff = [], []
    for i in range(N):
        for j in range(i + 1, N):
            cos = float(np.dot(e[i], e[j]))
            if labels[i] == labels[j]:
                same.append(cos)
            else:
                diff.append(cos)

    same_mean = np.mean(same) if same else 0.0
    diff_mean = np.mean(diff) if diff else 0.0
    separation = same_mean - diff_mean

    speakers = np.unique(labels)
    K = len(speakers)

    kmeans = KMeans(n_clusters=K, n_init=10, random_state=0)
    pred = kmeans.fit_predict(e)

    cluster_acc = _cluster_accuracy(pred, labels)
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


def _cluster_accuracy(pred_labels, true_labels):
    from collections import Counter
    pred = np.array(pred_labels)
    true = np.array(true_labels)
    total = 0

    for c in np.unique(pred):
        idx = pred == c
        true_subset = true[idx]
        if len(true_subset) == 0:
            continue
        total += Counter(true_subset).most_common(1)[0][1]

    return total / len(true)


# =====================================================================
# 5) TSNE visualization (same as your script)
# =====================================================================
def plot_tsne_subset(embs, labels, num_speakers=4, save_path="tsne_kmeans_subset.png"):
    speakers = np.unique(labels)
    if len(speakers) == 0:
        print("No speakers found, skipping TSNE.")
        return

    k = min(num_speakers, len(speakers))
    chosen = np.random.choice(speakers, size=k, replace=False)

    mask = np.isin(labels, chosen)
    X = embs[mask]
    Y = labels[mask]

    if X.shape[0] < 10:
        print("Too few points for TSNE, skipping.")
        return

    tsne = TSNE(
        n_components=2,
        perplexity=20,
        learning_rate="auto",
        init="pca",
        random_state=0,
    )
    coords = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    for spk in chosen:
        idx = (Y == spk)
        pts = coords[idx]
        plt.scatter(pts[:, 0], pts[:, 1], s=18, label=f"Spk {spk}")

    plt.legend(title="Speakers")
    plt.title("t-SNE of WavLM+KMeans Cluster Embeddings (subset of speakers)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[✔] Saved TSNE plot to {save_path}")


# =====================================================================
# 6) MAIN
# =====================================================================
if __name__ == "__main__":
    META = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/3sp/Libri3Mix_ovl50to80/wav16k/min/metadata/mixture_test_mix_clean.csv"
    TSNE_SAVE_PATH = "/home/sidharth./codebase/wavlm_dual_embedding/analysis/tsne_new/kmeans_devclean_whamtt_subset_tsne.png"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ---- Load Metadata ----
    metadata = parse_metadata(META)
    print(f"Loaded {len(metadata)} mixtures.")

    # ---- Load WavLM ----
    wavlm = load_wavlm(device=device)

    # ---- Extract WavLM+KMeans Embeddings ----
    embs, labels = extract_kmeans_embeddings_wavlm(
        wavlm=wavlm,
        metadata=metadata,
        device=device,
        verbose=True,
    )
    print(f"Extracted {len(embs)} embeddings for {len(np.unique(labels))} speakers.")

    # ---- Compute Metrics ----
    print("\nComputing clustering metrics (WavLM+KMeans)...")
    res = compute_clustering_metrics(embs, labels)

    print("\n=== Clustering / Separation Metrics (WavLM+KMeans, Full Dev) ===")
    print(f"same_mean_cos = {res['same_mean_cos']:.4f}")
    print(f"diff_mean_cos = {res['diff_mean_cos']:.4f}")
    print(f"separation    = {res['separation']:.4f}")
    print(f"cluster_acc   = {res['cluster_acc']:.4f}")
    print(f"nmi           = {res['nmi']:.4f}")
    print(f"ari           = {res['ari']:.4f}")
    print(f"silhouette    = {res['silhouette']:.4f}")

    # ---- TSNE on subset of speakers ----
    # plot_tsne_subset(embs, labels, num_speakers=4, save_path=TSNE_SAVE_PATH)