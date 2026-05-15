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


# ============================================================
# Hardcoded config
# ============================================================

META = "/home/sidcs/datasets/LibriMix/LibriMix/Libriuni_05_08/Libri2Mix_ovl50to80/wav16k/min/metadata/mixture_test_mix_clean.csv"

ONNX_PATH = "/home/sidcs/codebase/wavlm_dual_embedding/ecapa_feat_int8_static_conv_only.onnx"

TEACHER_CKPT = "/home/sidcs/model_ckpts/ecapa_tdnn_arcface_tr360/best-epoch=30-val_separation=0.000.ckpt"

TEACHER_C = 1024

RATE = 16000
CHUNK_SEC = 3.0
CHUNK_SAMPLES = int(CHUNK_SEC * RATE)

# Good for short LibriMix-style examples.
# The code uses up to L seconds, capped by actual file duration.
CONTEXT_LENGTHS = [3, 4.5, 6, 7.5, 9, 12, 15]

# Use -1 for all. Use e.g. 500 for quick testing.
MAX_ITEMS = -1

FRONTEND_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEACHER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 44

# Use evenly spaced chunks across available context.
USE_EVENLY_SPACED_CHUNKS = True

# Optional cap on number of 3-sec chunks.
# None means use natural number of chunks from effective context.
MAX_CHUNKS = None

# Save final summary as CSV too
SAVE_CSV = True
CSV_SAVE_PATH = "context_length_quantized_results.csv"


# ============================================================
# Imports from your codebase
# ============================================================

sys.path.append("/home/sidcs/codebase/")
from wavlm_single_embedding.model import ECAPA_TDNN as SingleSpeakerEncoderWrapper

sys.path.append("/home/sidcs/codebase/wavlm_dual_embedding")
from pvad_model import PreEmphasis


# ============================================================
# Frontend
# ============================================================

class LogMelFrontend(nn.Module):
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
        """
        wav: [B, T]
        returns: [B, 80, Frames]
        """
        x = self.torchfbank(wav) + 1e-6
        x = x.log()
        x = x - torch.mean(x, dim=-1, keepdim=True)
        return x


# ============================================================
# ONNX wrapper
# ============================================================

class QuantizedDualEmbeddingONNX:
    def __init__(self, onnx_path, frontend_device="cuda"):
        self.onnx_path = onnx_path
        self.frontend_device = torch.device(frontend_device)
        self.frontend = LogMelFrontend().to(self.frontend_device).eval()

        self.sess = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
        )

        print(f"[ONNX] Loaded: {onnx_path}")
        print("[ONNX] Providers:", self.sess.get_providers())

    @torch.no_grad()
    def wav_to_features_np(self, wav_1d_np):
        """
        wav_1d_np: [T]
        returns: [1, 80, Frames]
        """
        wav = torch.from_numpy(wav_1d_np.astype(np.float32)).unsqueeze(0)
        wav = wav.to(self.frontend_device)

        feat = self.frontend(wav)

        return feat.cpu().numpy().astype(np.float32)

    def run_chunk(self, chunk_np):
        """
        chunk_np: [48000] for 3 sec at 16 kHz

        returns:
            emb: torch [1, 2, 256]
        """
        feat_np = self.wav_to_features_np(chunk_np)

        emb_np = self.sess.run(
            ["embeddings"],
            {"features": feat_np.astype(np.float32)},
        )[0]

        emb = torch.from_numpy(emb_np).float()
        emb = F.normalize(emb, dim=-1)

        return emb

    def compute_context_embedding(self, wav_np, context_sec, max_chunks=None):
        """
        wav_np:
            full mixture waveform [T]

        context_sec:
            requested audio context length in seconds.

        Important:
            Uses effective context:
                effective_context = min(context_sec, actual_audio_duration)

            Does NOT pad to requested context length.
            Only pads if effective context is shorter than one 3-sec ONNX chunk.

        returns:
            emb_avg: torch [1, 2, 256]
            slot_cos: float
            num_chunks: int
            effective_context_sec: float
            actual_audio_sec: float
        """
        actual_audio_sec = len(wav_np) / RATE

        requested_samples = int(context_sec * RATE)
        effective_samples = min(len(wav_np), requested_samples)

        # Use only available audio up to requested context.
        context = wav_np[:effective_samples]

        # Only pad if shorter than one ONNX chunk.
        if len(context) < CHUNK_SAMPLES:
            context = np.pad(context, (0, CHUNK_SAMPLES - len(context)))

        effective_context_sec = min(actual_audio_sec, context_sec)

        max_start = max(0, len(context) - CHUNK_SAMPLES)

        # Natural number of 3-sec chunks available in effective context.
        # Examples:
        #   3.0 sec  -> 1 chunk
        #   4.5 sec  -> 2 possible coverage chunks if evenly spaced
        #   9.0 sec  -> 3 chunks
        natural_chunks = max(1, int(np.ceil(effective_context_sec / CHUNK_SEC)))

        if max_chunks is None:
            num_chunks = natural_chunks
        else:
            num_chunks = min(max_chunks, natural_chunks)

        if effective_context_sec <= CHUNK_SEC or max_start == 0:
            starts = [0]
        else:
            if USE_EVENLY_SPACED_CHUNKS:
                if num_chunks <= 1:
                    starts = [max_start]
                else:
                    starts = np.linspace(0, max_start, num_chunks).astype(int).tolist()
            else:
                starts = list(range(0, max_start + 1, CHUNK_SAMPLES))
                starts = starts[:num_chunks]

        embs = []

        for st in starts:
            chunk = context[st:st + CHUNK_SAMPLES]

            if len(chunk) < CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))

            emb = self.run_chunk(chunk)
            embs.append(emb)

        embs = torch.cat(embs, dim=0)  # [N, 2, 256]

        emb_avg = embs.mean(dim=0, keepdim=True)
        emb_avg = F.normalize(emb_avg, dim=-1)

        slot_cos = F.cosine_similarity(
            emb_avg[:, 0, :],
            emb_avg[:, 1, :],
            dim=-1,
        ).item()

        return emb_avg, slot_cos, len(starts), effective_context_sec, actual_audio_sec


# ============================================================
# Metadata
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
# Teacher
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

    missing, unexpected = teacher.load_state_dict(filtered, strict=False)

    print(f"[Teacher] missing={len(missing)} unexpected={len(unexpected)}")

    if len(missing) > 0:
        print("[Teacher] Missing examples:")
        for k in missing[:10]:
            print(" ", k)

    if len(unexpected) > 0:
        print("[Teacher] Unexpected examples:")
        for k in unexpected[:10]:
            print(" ", k)

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

    e = teacher_model(wav)
    e = F.normalize(e, dim=-1)

    return e.squeeze(0)


# ============================================================
# Load waveform
# ============================================================

def load_full_wav_np(path):
    wav, sr = torchaudio.load(path)

    if sr != RATE:
        wav = torchaudio.functional.resample(wav, sr, RATE)

    wav = wav.mean(0).cpu().numpy().astype(np.float32)

    return wav


# ============================================================
# Metrics
# ============================================================

def compute_clustering_metrics(embs, labels):
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
    e = embs / norms
    N = e.shape[0]

    same = []
    diff = []

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
# Evaluation for one context length
# ============================================================

def evaluate_context_length(
    quant_model,
    teacher_model,
    metadata,
    context_sec,
    teacher_device="cuda",
):
    all_embs = []
    all_labels = []

    slot_cosines = []
    num_chunks_list = []
    matched_student_teacher_cos = []

    effective_context_list = []
    actual_audio_duration_list = []

    def cosine(a, b):
        return (a @ b) / (a.norm() * b.norm() + 1e-8)

    iterator = tqdm(metadata, desc=f"Context requested {context_sec}s")

    for entry in iterator:
        wav_np = load_full_wav_np(entry["mix_path"])

        (
            ed,
            slot_cos,
            num_chunks,
            effective_context_sec,
            actual_audio_sec,
        ) = quant_model.compute_context_embedding(
            wav_np,
            context_sec=context_sec,
            max_chunks=MAX_CHUNKS,
        )

        ed = ed.squeeze(0)  # [2, 256]
        e0 = ed[0]
        e1 = ed[1]

        slot_cosines.append(slot_cos)
        num_chunks_list.append(num_chunks)
        effective_context_list.append(effective_context_sec)
        actual_audio_duration_list.append(actual_audio_sec)

        t1 = get_teacher_emb(teacher_model, entry["src1"], teacher_device)
        t2 = get_teacher_emb(teacher_model, entry["src2"], teacher_device)

        e0_d = e0.to(teacher_device)
        e1_d = e1.to(teacher_device)

        score_direct = cosine(e0_d, t1) + cosine(e1_d, t2)
        score_swap = cosine(e0_d, t2) + cosine(e1_d, t1)

        if score_direct >= score_swap:
            mapped = [
                (e0, entry["spk1"], cosine(e0_d, t1).item()),
                (e1, entry["spk2"], cosine(e1_d, t2).item()),
            ]
        else:
            mapped = [
                (e0, entry["spk2"], cosine(e0_d, t2).item()),
                (e1, entry["spk1"], cosine(e1_d, t1).item()),
            ]

        for e, lab, st_cos in mapped:
            all_embs.append(e.cpu().numpy())
            all_labels.append(lab)
            matched_student_teacher_cos.append(st_cos)

    embs = np.vstack(all_embs)
    labels = np.array(all_labels)

    slot_cosines = np.array(slot_cosines)
    matched_student_teacher_cos = np.array(matched_student_teacher_cos)
    effective_context_list = np.array(effective_context_list)
    actual_audio_duration_list = np.array(actual_audio_duration_list)
    num_chunks_list = np.array(num_chunks_list)

    metrics = compute_clustering_metrics(embs, labels)

    diagnostics = {
        "requested_context_sec": float(context_sec),
        "avg_effective_context_sec": float(np.mean(effective_context_list)),
        "min_effective_context_sec": float(np.min(effective_context_list)),
        "max_effective_context_sec": float(np.max(effective_context_list)),
        "avg_actual_audio_sec": float(np.mean(actual_audio_duration_list)),
        "min_actual_audio_sec": float(np.min(actual_audio_duration_list)),
        "max_actual_audio_sec": float(np.max(actual_audio_duration_list)),
        "avg_num_chunks": float(np.mean(num_chunks_list)),
        "min_num_chunks": float(np.min(num_chunks_list)),
        "max_num_chunks": float(np.max(num_chunks_list)),
        "slot_cos_mean": float(slot_cosines.mean()),
        "slot_cos_std": float(slot_cosines.std()),
        "slot_cos_gt_075": float((slot_cosines > 0.75).mean()),
        "slot_cos_gt_085": float((slot_cosines > 0.85).mean()),
        "matched_st_cos_mean": float(matched_student_teacher_cos.mean()),
        "matched_st_cos_std": float(matched_student_teacher_cos.std()),
    }

    return metrics, diagnostics


# ============================================================
# Save CSV
# ============================================================

def save_rows_to_csv(rows, path):
    import csv

    if len(rows) == 0:
        return

    keys = list(rows[0].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(f"\n[Saved] CSV results: {path}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Frontend device:", FRONTEND_DEVICE)
    print("Teacher device:", TEACHER_DEVICE)
    print("Metadata:", META)
    print("ONNX:", ONNX_PATH)
    print("Context lengths:", CONTEXT_LENGTHS)
    print("MAX_ITEMS:", MAX_ITEMS)

    metadata = parse_metadata(META)

    if MAX_ITEMS > 0:
        metadata = metadata[:MAX_ITEMS]

    print(f"Loaded {len(metadata)} mixtures.")

    teacher = load_teacher_model(
        TEACHER_CKPT,
        device=TEACHER_DEVICE,
        C=TEACHER_C,
    )

    quant_model = QuantizedDualEmbeddingONNX(
        ONNX_PATH,
        frontend_device=FRONTEND_DEVICE,
    )

    all_rows = []

    for context_sec in CONTEXT_LENGTHS:
        metrics, diag = evaluate_context_length(
            quant_model=quant_model,
            teacher_model=teacher,
            metadata=metadata,
            context_sec=context_sec,
            teacher_device=TEACHER_DEVICE,
        )

        row = {}
        row.update(diag)
        row.update(metrics)
        all_rows.append(row)

        print("\n==================================================")
        print(f"Requested context length: {context_sec}s")
        print("==================================================")
        print(f"avg_effective_context_sec = {row['avg_effective_context_sec']:.2f}")
        print(f"min_effective_context_sec = {row['min_effective_context_sec']:.2f}")
        print(f"max_effective_context_sec = {row['max_effective_context_sec']:.2f}")
        print(f"avg_actual_audio_sec      = {row['avg_actual_audio_sec']:.2f}")
        print(f"avg_num_chunks            = {row['avg_num_chunks']:.2f}")
        print(f"matched_ST_cos_mean       = {row['matched_st_cos_mean']:.4f}")
        print(f"slot_cos_mean             = {row['slot_cos_mean']:.4f}")
        print(f"slot_cos > 0.75           = {row['slot_cos_gt_075']:.4f}")
        print(f"same_mean_cos             = {row['same_mean_cos']:.4f}")
        print(f"diff_mean_cos             = {row['diff_mean_cos']:.4f}")
        print(f"separation                = {row['separation']:.4f}")
        print(f"cluster_acc               = {row['cluster_acc']:.4f}")
        print(f"nmi                       = {row['nmi']:.4f}")
        print(f"ari                       = {row['ari']:.4f}")
        print(f"silhouette                = {row['silhouette']:.4f}")

    print("\n\n================ FINAL SUMMARY ================")
    header = (
        "ReqL | EffL | ActL | chunks | STcos | slotcos | slot>0.75 | "
        "same | diff | sep | acc | nmi | ari | sil"
    )

    print(header)
    print("-" * len(header))

    for row in all_rows:
        print(
            f"{row['requested_context_sec']:4.1f} | "
            f"{row['avg_effective_context_sec']:4.1f} | "
            f"{row['avg_actual_audio_sec']:4.1f} | "
            f"{row['avg_num_chunks']:6.2f} | "
            f"{row['matched_st_cos_mean']:5.3f} | "
            f"{row['slot_cos_mean']:7.3f} | "
            f"{row['slot_cos_gt_075']:9.3f} | "
            f"{row['same_mean_cos']:5.3f} | "
            f"{row['diff_mean_cos']:5.3f} | "
            f"{row['separation']:5.3f} | "
            f"{row['cluster_acc']:5.3f} | "
            f"{row['nmi']:5.3f} | "
            f"{row['ari']:5.3f} | "
            f"{row['silhouette']:5.3f}"
        )

    if SAVE_CSV:
        save_rows_to_csv(all_rows, CSV_SAVE_PATH)