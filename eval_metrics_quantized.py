import os
import sys
import random

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnxruntime as ort

from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
)

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# ============================================================
# Hardcoded config
# ============================================================

META = "/home/sidcs/datasets/LibriMix/LibriMix/Libriuni_05_08/Libri2Mix_ovl50to80/wav16k/min/metadata/mixture_test_mix_clean.csv"

# ONNX_PATH = "/home/sidcs/codebase/wavlm_dual_embedding/ecapa_feat_int8_static_conv_only.onnx" #int8
ONNX_PATH = "/home/sidcs/codebase/wavlm_dual_embedding/ecapa_feat_fp32.onnx" #fp32

TEACHER_CKPT = "/home/sidcs/model_ckpts/ecapa_tdnn_arcface_tr360/best-epoch=30-val_separation=0.000.ckpt"

TEACHER_C = 1024

SECONDS = 3.0  # because ONNX was exported for ~3 sec, [1, 80, 301]

FRONTEND_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEACHER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ADD_NOISE = False
NOISE_DIR = "/home/sidcs/datasets/LibriMix/LibriMix/wham_noise/tt"

MAX_ITEMS = -1  # use -1 for all, e.g. 200 for quick test

SEED = 44

DO_TSNE = False
TSNE_SPEAKERS = 40
TSNE_SAVE = "tsne_quantized_dual_40sp.png"

RATE = 16000
EPS = 1e-10


# ============================================================
# Imports from your codebase
# ============================================================

sys.path.append("/home/sidcs/codebase/")
from wavlm_single_embedding.model import ECAPA_TDNN as SingleSpeakerEncoderWrapper

sys.path.append("/home/sidcs/codebase/wavlm_dual_embedding")
from pvad_model import PreEmphasis


# ============================================================
# Log-Mel frontend used before quantized ECAPA ONNX
# ============================================================

class LogMelFrontend(nn.Module):
    """
    Input:
        wav: [B, T]

    Output:
        logmel: [B, 80, Frames]
    """
    def __init__(self):
        super().__init__()

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=512,
                win_length=400,
                hop_length=160,
                f_min=20,
                f_max=7600,
                window_fn=torch.hamming_window,
                n_mels=80,
            ),
        )

    @torch.no_grad()
    def forward(self, wav):
        x = self.torchfbank(wav) + 1e-6
        x = x.log()
        x = x - torch.mean(x, dim=-1, keepdim=True)
        return x


# ============================================================
# Quantized ONNX dual embedding wrapper
# ============================================================

class QuantizedDualEmbeddingONNX:
    """
    Quantized ECAPA ONNX model.

    Expected ONNX input:
        features: [1, 80, T_frames]

    Expected ONNX output:
        embeddings: [1, 2, 256]
    """
    def __init__(
        self,
        onnx_path,
        frontend_device="cuda",
        chunk_sec=3.0,
        providers=None,
    ):
        self.onnx_path = onnx_path
        self.frontend_device = torch.device(frontend_device)
        self.chunk_sec = float(chunk_sec)
        self.chunk_samples = int(self.chunk_sec * RATE)

        self.frontend = LogMelFrontend().to(self.frontend_device).eval()

        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.sess = ort.InferenceSession(
            onnx_path,
            providers=providers,
        )

        print(f"[ONNX] Loaded: {onnx_path}")
        print("[ONNX] Providers:", self.sess.get_providers())

    @torch.no_grad()
    def wav_to_features_np(self, wav):
        """
        wav: torch [1, T] or numpy [T]
        returns: numpy [1, 80, Frames]
        """
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)

        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        wav = wav.to(self.frontend_device).float()
        feat = self.frontend(wav)

        return feat.cpu().numpy().astype(np.float32)

    def run_features(self, feat_np):
        emb_np = self.sess.run(
            ["embeddings"],
            {"features": feat_np.astype(np.float32)},
        )[0]

        return emb_np.astype(np.float32)

    def __call__(self, wav):
        """
        wav: torch [1, T]

        returns:
            emb: torch [1, 2, 256]
        """
        feat_np = self.wav_to_features_np(wav)
        emb_np = self.run_features(feat_np)

        emb = torch.from_numpy(emb_np).float()
        emb = F.normalize(emb, dim=-1)

        return emb


# ============================================================
# Optional noise
# ============================================================

def collect_noise_files(noise_dir):
    if noise_dir is None or not os.path.isdir(noise_dir):
        return []

    return [
        os.path.join(noise_dir, f)
        for f in os.listdir(noise_dir)
        if f.endswith(".wav")
    ]


def mix_with_snr(clean, noise, snr_db):
    """
    clean, noise: torch [T]
    """
    if noise.ndim > 1:
        noise = noise.mean(0)

    if len(noise) < len(clean):
        diff = len(clean) - len(noise)
        noise = F.pad(noise, (diff // 2, diff - diff // 2))
    else:
        noise = noise[:len(clean)]

    clean_power = clean.pow(2).mean()
    noise_power = noise.pow(2).mean()

    target_noise_power = clean_power / (10 ** (snr_db / 10))
    scale = torch.sqrt(target_noise_power / (noise_power + 1e-8))

    return clean + scale * noise


# ============================================================
# Metadata parser
# ============================================================

def parse_metadata(csv_path):
    metadata = []
    filename = os.path.basename(csv_path)

    with open(csv_path, "r") as f:
        _ = next(f)

        for line in f:
            parts = line.strip().split(",")

            if len(parts) == 0:
                continue

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

            metadata.append(
                {
                    "mix_id": mix_id,
                    "mix_path": mix_path,
                    "src1": src1,
                    "src2": src2,
                    "spk1": int(spk1),
                    "spk2": int(spk2),
                }
            )

    return metadata


# ============================================================
# Teacher loading
# ============================================================

def load_teacher_model(teacher_ckpt, device="cuda", C=1024):
    teacher = SingleSpeakerEncoderWrapper(C=C).to(device)

    ckpt = torch.load(teacher_ckpt, map_location=device)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    filtered = {}

    for k, v in state.items():
        if not k.startswith("model."):
            continue

        if "arcface" in k or "arc_face" in k:
            continue

        new_k = k.replace("model.", "", 1)
        filtered[new_k] = v

    print("[Teacher] Loaded keys:", len(filtered))

    teacher.load_state_dict(filtered, strict=True)
    teacher.eval()

    for p in teacher.parameters():
        p.requires_grad = False

    return teacher


@torch.no_grad()
def get_teacher_emb(teacher_model, wav_path, device="cuda"):
    wav, sr = torchaudio.load(wav_path)

    if sr != RATE:
        wav = torchaudio.functional.resample(wav, sr, RATE)

    wav = wav.mean(0).to(device).unsqueeze(0)

    e = teacher_model(wav)  # [1, 256]
    e = F.normalize(e, dim=-1)

    return e.squeeze(0)  # [256]


# ============================================================
# Audio loading for ONNX model
# ============================================================

def load_mix_audio(
    mix_path,
    seconds=3.0,
    add_noise=False,
    noise_files=None,
):
    wav, sr = torchaudio.load(mix_path)

    if sr != RATE:
        wav = torchaudio.functional.resample(wav, sr, RATE)

    wav = wav.mean(0)

    if seconds is not None and seconds > 0:
        num_samples = int(seconds * RATE)

        if wav.numel() < num_samples:
            wav = F.pad(wav, (0, num_samples - wav.numel()))
        else:
            wav = wav[:num_samples]

    if add_noise and noise_files:
        noise_p = random.choice(noise_files)
        noise_wav, noise_sr = torchaudio.load(noise_p)

        if noise_sr != RATE:
            noise_wav = torchaudio.functional.resample(noise_wav, noise_sr, RATE)

        noise_wav = noise_wav.squeeze(0)

        r = random.random()
        if r < 0.4:
            snr = random.uniform(-5, 5)
        elif r < 0.8:
            snr = random.uniform(5, 15)
        else:
            snr = random.uniform(15, 25)

        wav = mix_with_snr(wav, noise_wav, snr)

    return wav.unsqueeze(0)  # [1, T]


# ============================================================
# PIT teacher-aligned extraction for quantized model
# ============================================================

def extract_quantized_dual_embeddings_with_teacher(
    quant_dual_model,
    teacher_model,
    metadata,
    device="cuda",
    seconds=3.0,
    add_noise=False,
    noise_files=None,
    verbose=True,
):
    all_embs = []
    all_labels = []
    slot_cosines = []

    def cosine(a, b):
        return (a @ b) / (a.norm() * b.norm() + 1e-8)

    iterator = tqdm(
        metadata,
        desc="Extracting quantized ONNX embeddings",
        disable=not verbose,
    )

    for entry in iterator:
        mix_path = entry["mix_path"]
        src1 = entry["src1"]
        src2 = entry["src2"]
        spk1 = entry["spk1"]
        spk2 = entry["spk2"]

        # --------------------------
        # Load mixture for ONNX model
        # --------------------------
        mix = load_mix_audio(
            mix_path=mix_path,
            seconds=seconds,
            add_noise=add_noise,
            noise_files=noise_files,
        )  # [1, T], CPU

        # --------------------------
        # Quantized dual embeddings
        # --------------------------
        with torch.no_grad():
            ed = quant_dual_model(mix)  # [1, 2, 256]

        ed = ed.squeeze(0)  # [2, 256]
        e0 = ed[0]
        e1 = ed[1]

        slot_cos = F.cosine_similarity(
            e0.unsqueeze(0),
            e1.unsqueeze(0),
            dim=-1,
        ).item()
        slot_cosines.append(slot_cos)

        # --------------------------
        # Teacher embeddings from clean sources
        # --------------------------
        t1 = get_teacher_emb(teacher_model, src1, device)
        t2 = get_teacher_emb(teacher_model, src2, device)

        e0_d = e0.to(device)
        e1_d = e1.to(device)

        # --------------------------
        # PIT matching
        # --------------------------
        score_direct = cosine(e0_d, t1) + cosine(e1_d, t2)
        score_swap = cosine(e0_d, t2) + cosine(e1_d, t1)

        if score_direct >= score_swap:
            mapped = [(e0, spk1), (e1, spk2)]
        else:
            mapped = [(e0, spk2), (e1, spk1)]

        for e, lab in mapped:
            all_embs.append(e.cpu().numpy())
            all_labels.append(lab)

    return np.vstack(all_embs), np.array(all_labels), np.array(slot_cosines)


# ============================================================
# Metrics
# ============================================================

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


# ============================================================
# Optional TSNE
# ============================================================

def plot_tsne_subset(embs, labels, num_speakers=40, save_path="tsne_quantized_subset.png"):
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

    perplexity = min(20, max(5, X.shape[0] // 5))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=0,
    )

    coords = tsne.fit_transform(X)

    cmap = plt.cm.get_cmap("turbo", k)
    markers = ["o", "^", "s", "D", "v", "P", "X", "*", "<", ">", "h", "H", "p", "+", "x"]
    m = len(markers)

    plt.figure(figsize=(10, 8))

    for i, spk in enumerate(chosen):
        idx = Y == spk
        pts = coords[idx]

        plt.scatter(
            pts[:, 0],
            pts[:, 1],
            s=22,
            c=[cmap(i)],
            marker=markers[i % m],
            alpha=0.80,
            linewidths=0.3,
            edgecolors="k",
            label=f"Spk {spk}",
        )

    plt.legend(
        title="Speakers",
        ncol=2,
        fontsize=7,
        title_fontsize=8,
        markerscale=1.0,
        frameon=True,
    )

    plt.title("t-SNE of Quantized Dual Speaker Embeddings")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[✔] Saved TSNE plot to {save_path}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Frontend device:", FRONTEND_DEVICE)
    print("Teacher device:", TEACHER_DEVICE)
    print("Meta:", META)
    print("ONNX:", ONNX_PATH)
    print("Teacher checkpoint:", TEACHER_CKPT)
    print("Seconds:", SECONDS)

    metadata = parse_metadata(META)

    if MAX_ITEMS > 0:
        metadata = metadata[:MAX_ITEMS]

    print(f"Loaded {len(metadata)} mixtures.")

    noise_files = collect_noise_files(NOISE_DIR) if ADD_NOISE else []

    if ADD_NOISE:
        print("Noise enabled.")
        print("Noise files:", len(noise_files))

    teacher = load_teacher_model(
        teacher_ckpt=TEACHER_CKPT,
        device=TEACHER_DEVICE,
        C=TEACHER_C,
    )

    quant_dual = QuantizedDualEmbeddingONNX(
        onnx_path=ONNX_PATH,
        frontend_device=FRONTEND_DEVICE,
        chunk_sec=SECONDS,
        providers=["CPUExecutionProvider"],
    )

    embs, labels, slot_cosines = extract_quantized_dual_embeddings_with_teacher(
        quant_dual_model=quant_dual,
        teacher_model=teacher,
        metadata=metadata,
        device=TEACHER_DEVICE,
        seconds=SECONDS,
        add_noise=ADD_NOISE,
        noise_files=noise_files,
        verbose=True,
    )

    print(f"Extracted {len(embs)} embeddings for {len(np.unique(labels))} speakers.")

    print("\n=== Slot cosine diagnostics ===")
    print(f"slot_cos mean = {slot_cosines.mean():.4f}")
    print(f"slot_cos min  = {slot_cosines.min():.4f}")
    print(f"slot_cos max  = {slot_cosines.max():.4f}")
    print(f"slot_cos > 0.75 = {(slot_cosines > 0.75).mean():.4f}")
    print(f"slot_cos > 0.85 = {(slot_cosines > 0.85).mean():.4f}")

    print("\nComputing clustering metrics...")
    res = compute_clustering_metrics(embs, labels)

    print("\n=== Clustering / Separation Metrics: Quantized ONNX Dual Model ===")
    print(f"same_mean_cos = {res['same_mean_cos']:.4f}")
    print(f"diff_mean_cos = {res['diff_mean_cos']:.4f}")
    print(f"separation    = {res['separation']:.4f}")
    print(f"cluster_acc   = {res['cluster_acc']:.4f}")
    print(f"nmi           = {res['nmi']:.4f}")
    print(f"ari           = {res['ari']:.4f}")
    print(f"silhouette    = {res['silhouette']:.4f}")

    if DO_TSNE:
        plot_tsne_subset(
            embs,
            labels,
            num_speakers=TSNE_SPEAKERS,
            save_path=TSNE_SAVE,
        )