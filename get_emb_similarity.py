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

sys.path.append('/home/sidcs/codebase/wavlm_dual_embedding')
from model import SpeakerEncoderDualWrapper   # your dual model class

# Needed for teacher model
sys.path.append("/home/sidcs/codebase/")
# from wavlm_single_embedding.model import SpeakerEncoderWrapper as SingleSpkEncoder
from wavlm_single_embedding.model import ECAPA_TDNN as SingleSpeakerEncoderWrapper


# -----------------------------
# Global noise config
# -----------------------------
add_noise = True  # set True to enable noise corruption
noise_dir = "/home/sidcs/datasets/LibriMix/LibriMix/wham_noise/tt"
noise_files = [
    os.path.join(noise_dir, f)
    for f in os.listdir(noise_dir)
    if f.endswith(".wav")
]

random.seed(44)


# =====================================================================
# 0) Utility: mix with SNR
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
# 1) Clean dual-model weight loading
# =====================================================================
def joint_trained_model_weights(state):
    new_state = {}
    for k, v in state.items():
        if k.startswith('dual_emb_model.'):
            k2 = k.replace('dual_emb_model.', '')
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


def load_dual_model(ckpt_path, emb_dim=256, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = strip_dual_model_weights(ckpt["state_dict"])
    model = SpeakerEncoderDualWrapper(emb_dim=emb_dim)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


# =====================================================================
# 2) LibriMix metadata parser
# =====================================================================
def parse_metadata(csv_path):
    metadata = []
    filename = os.path.basename(csv_path)

    with open(csv_path, "r") as f:
        _ = next(f)  # header
        for line in f:
            parts = line.strip().split(",")

            if filename.split("_")[-1] == "both.csv":
                if len(parts) != 8:
                    continue
                mix_id, mix_path, src1, src2, spk1, spk2, noise, length = parts
            else:
                if len(parts) != 7:
                    continue
                mix_id, mix_path, src1, src2, spk1, spk2, length = parts

            if spk1 == "speaker_1_ID":
                continue

            metadata.append({
                "mix_path": mix_path,
                "src1": src1,
                "src2": src2,
                "spk1": int(spk1),
                "spk2": int(spk2),
            })

    return metadata


# =====================================================================
# 3) Teacher embedding extraction
# =====================================================================
def get_teacher_emb(teacher_model, wav_path, device="cuda"):
    wav, sr = torchaudio.load(wav_path)
    wav = wav.mean(0).to(device).unsqueeze(0)

    with torch.no_grad():
        e = teacher_model(wav)  # [1,256]
    return e.squeeze(0)  # [256]


# =====================================================================
# Helper: cosine
# =====================================================================
def cosine(a, b):
    return (a @ b) / (a.norm() * b.norm() + 1e-8)


def _stats(x: np.ndarray):
    if x.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(x.mean()),
        "std": float(x.std(ddof=0)),
        "n": int(x.size),
    }


# =====================================================================
# 3b) Dual embedding extraction + PIT + cosine stats
# =====================================================================
def extract_dual_embeddings_with_teacher(
    dual_model,
    teacher_model,
    metadata,
    device="cuda",
    verbose=True
):
    """
    Returns:
      all_embs   -> [2*num_mixtures, D]
      all_labels -> [2*num_mixtures]
      cos_stats  -> dict with:
        - matched_student_teacher: cos(ê_i, t_i) after PIT (2 per mixture)
        - student_pair: cos(e0, e1) per mixture (1 per mixture)
        - teacher_pair: cos(t1, t2) per mixture (1 per mixture)
    """
    all_embs = []
    all_labels = []

    matched_cos = []         # 2 per mixture (after PIT)
    student_pair_cos = []    # 1 per mixture: cos(e0, e1)
    teacher_pair_cos = []    # 1 per mixture: cos(t1, t2)

    iterator = tqdm(metadata, desc="Extracting embeddings", disable=not verbose)

    for entry in iterator:
        mix_path = entry["mix_path"]
        src1 = entry["src1"]
        src2 = entry["src2"]
        spk1 = entry["spk1"]
        spk2 = entry["spk2"]

        # ------------ Load mixture ------------
        mix, sr = torchaudio.load(mix_path)
        mix = mix.mean(0)

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

        mix = mix.to(device).unsqueeze(0)

        # ------------ Dual embeddings ------------
        with torch.no_grad():
            ed = dual_model(mix)
        e0, e1 = ed.squeeze(0)   # [2,256]

        # Student-pair similarity (per mixture)
        student_pair_cos.append(float(cosine(e0, e1)))

        # ------------ Teacher embeddings ------------
        t1 = get_teacher_emb(teacher_model, src1, device)
        t2 = get_teacher_emb(teacher_model, src2, device)

        # Teacher-pair similarity (per mixture)
        teacher_pair_cos.append(float(cosine(t1, t2)))

        # ------------ PIT matching + matched similarity ------------
        s00 = cosine(e0, t1)  # e0 -> t1
        s11 = cosine(e1, t2)  # e1 -> t2
        s01 = cosine(e0, t2)  # e0 -> t2
        s10 = cosine(e1, t1)  # e1 -> t1

        score_direct = s00 + s11
        score_swap   = s01 + s10

        if score_direct >= score_swap:
            mapped = [(e0, spk1), (e1, spk2)]
            matched_cos.append(float(s00))
            matched_cos.append(float(s11))
        else:
            mapped = [(e0, spk2), (e1, spk1)]
            matched_cos.append(float(s01))
            matched_cos.append(float(s10))

        # ------------ Store ------------
        for e, lab in mapped:
            all_embs.append(e.cpu().numpy())
            all_labels.append(lab)

    matched_cos = np.asarray(matched_cos, dtype=np.float32)
    student_pair_cos = np.asarray(student_pair_cos, dtype=np.float32)
    teacher_pair_cos = np.asarray(teacher_pair_cos, dtype=np.float32)

    cos_stats = {
        "matched_student_teacher": _stats(matched_cos),
        "student_pair": _stats(student_pair_cos),
        "teacher_pair": _stats(teacher_pair_cos),
    }

    return np.vstack(all_embs), np.array(all_labels), cos_stats


# =====================================================================
# 4) Clustering + Separation Metrics
# =====================================================================
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


def compute_clustering_metrics(embs, labels):
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
    e = embs / norms
    N = e.shape[0]

    same, diff = [], []
    for i in range(N):
        for j in range(i + 1, N):
            cosv = float(np.dot(e[i], e[j]))
            if labels[i] == labels[j]:
                same.append(cosv)
            else:
                diff.append(cosv)

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


# =====================================================================
# 5) TSNE visualization (optional)
# =====================================================================
def plot_tsne_subset(embs, labels, num_speakers=40, save_path="tsne_subset.png"):
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
        random_state=0
    )
    coords = tsne.fit_transform(X)

    cmap = plt.cm.get_cmap("turbo", k)
    markers = ['o', '^', 's', 'D', 'v', 'P', 'X', '*', '<', '>', 'h', 'H', 'p', '+', 'x', '1', '2', '3', '4']
    m = len(markers)

    styles = {}
    for i, spk in enumerate(chosen):
        styles[spk] = {"color": cmap(i), "marker": markers[i % m]}

    plt.figure(figsize=(10, 8))
    for spk in chosen:
        idx = (Y == spk)
        pts = coords[idx]
        st = styles[spk]
        plt.scatter(
            pts[:, 0], pts[:, 1],
            s=22,
            c=[st["color"]],
            marker=st["marker"],
            alpha=0.80,
            linewidths=0.3,
            edgecolors="k",
            label=f"Spk {spk}"
        )

    plt.legend(
        title="Speakers",
        ncol=2,
        fontsize=7,
        title_fontsize=8,
        markerscale=1.0,
        frameon=True
    )
    plt.title("t-SNE of Dual Speaker Embeddings (subset of speakers)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[✔] Saved TSNE plot to {save_path}")


# =====================================================================
# 6) MAIN
# =====================================================================
if __name__ == "__main__":
    META = "/home/sidcs/datasets/LibriMix/LibriMix/Libriuni_05_08/Libri2Mix_ovl50to80/wav16k/min/metadata/mixture_test_mix_clean.csv"

    # CKPT = "/home/sidcs/model_ckpts/ECAPA_UNMIX_2048_teacher_ECAPA/best-epoch=117-val_separation=0.000.ckpt"
    # CKPT = "/home/sidcs/model_ckpts/librispeech_asp_ft_ecapa_linear_dualemb_tr360/best-epoch=144-val_separation=0.000.ckpt"
    CKPT = "/home/sidcs/model_ckpts/ECAPA_UNMIX_3072_teacher_ECAPA/best-epoch=141-val_separation=0.000.ckpt"
    TEACHER_CKPT = "/home/sidcs/model_ckpts/ecapa_tdnn_arcface_tr360/best-epoch=30-val_separation=0.000.ckpt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ---- Load Metadata ----
    metadata = parse_metadata(META)
    print(f"Loaded {len(metadata)} mixtures.")

    # ---- Load Teacher Model ----
    teacher = SingleSpeakerEncoderWrapper(C=1024).to(device)
    ckpt = torch.load(TEACHER_CKPT, map_location=device)
    state = ckpt["state_dict"]

    filtered = {}
    for k, v in state.items():
        if not k.startswith("model."):
            continue
        if "arcface" in k or "arc_face" in k:
            continue
        new_k = k.replace("model.", "", 1)
        filtered[new_k] = v

    print("Loaded teacher keys:", len(filtered))
    teacher.load_state_dict(filtered, strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # ---- Load Dual Model ----
    dual = load_dual_model(CKPT, device=device)
    breakpoint()

    # ---- Extract Embeddings + Cosine Stats ----
    embs, labels, cos_stats = extract_dual_embeddings_with_teacher(
        dual_model=dual,
        teacher_model=teacher,
        metadata=metadata,
        device=device,
        verbose=True,
    )
    print(f"Extracted {len(embs)} embeddings for {len(np.unique(labels))} speakers.")

    print("\n=== Cosine Similarity Stats ===")
    ms = cos_stats["matched_student_teacher"]
    sp = cos_stats["student_pair"]
    tp = cos_stats["teacher_pair"]

    print(f"Matched student↔teacher (after PIT): mean={ms['mean']:.4f}, std={ms['std']:.4f}, n={ms['n']}")
    print(f"Student pair e0↔e1 (per mixture):     mean={sp['mean']:.4f}, std={sp['std']:.4f}, n={sp['n']}")
    print(f"Teacher pair t1↔t2 (per mixture):     mean={tp['mean']:.4f}, std={tp['std']:.4f}, n={tp['n']}")

    # ---- (Optional) clustering metrics ----
    print(f"\nComputing clustering metrics...")
    res = compute_clustering_metrics(embs, labels)
    print("\n=== Clustering / Separation Metrics (Full Dev) ===")
    print(f"same_mean_cos = {res['same_mean_cos']:.4f}")
    print(f"diff_mean_cos = {res['diff_mean_cos']:.4f}")
    print(f"separation    = {res['separation']:.4f}")
    print(f"cluster_acc   = {res['cluster_acc']:.4f}")
    print(f"nmi           = {res['nmi']:.4f}")
    print(f"ari           = {res['ari']:.4f}")
    print(f"silhouette    = {res['silhouette']:.4f}")

    # Optional TSNE:
    plot_tsne_subset(embs, labels, num_speakers=20, save_path="/home/sidcs/codebase/wavlm_dual_embedding/analysis/tsne_ecapa/tsne_subset_ecapa3072.png")