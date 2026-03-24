import os
import sys
import random
import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import json
import multiprocessing as mp
# -----------------------------
# Your imports
# -----------------------------
sys.path.append('/home/sidharth./codebase/wavlm_dual_embedding')
from model import SpeakerEncoderDualWrapper

sys.path.append("/home/sidharth./codebase/")
from wavlm_single_embedding.model import SpeakerEncoderWrapper as SingleSpkEncoder


# -----------------------------
# Global noise config
# -----------------------------
add_noise = True
noise_dir = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/wham_noise/tt"
noise_files = [
    os.path.join(noise_dir, f)
    for f in os.listdir(noise_dir)
    if f.endswith(".wav")
]

# -----------------------------
# Reproducibility
# -----------------------------
SEED = 44
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# =====================================================================
# 0) Utility: mix with SNR
# =====================================================================
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


# =====================================================================
# 1) Weight loading helpers
# =====================================================================
def joint_trained_model_weights(state):
    new_state = {}
    for k, v in state.items():
        if k.startswith('dual_emb_model.'):
            new_state[k.replace('dual_emb_model.', '')] = v
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
# 3) Teacher-aligned dual embedding extraction (PIT-based)
# =====================================================================
def get_teacher_emb(teacher_model, wav_path, device="cuda"):
    wav, sr = torchaudio.load(wav_path)
    wav = wav.mean(0).to(device).unsqueeze(0)
    with torch.no_grad():
        e = teacher_model(wav)  # [1,256]
    return e.squeeze(0)

def extract_dual_embeddings_with_teacher(
    dual_model,
    teacher_model,
    metadata,
    device="cuda",
    verbose=True
):
    all_embs = []
    all_labels = []

    def cosine(a, b):
        return (a @ b) / (a.norm() * b.norm() + 1e-8)

    iterator = tqdm(metadata, desc="Extracting embeddings", disable=not verbose)

    for entry in iterator:
        mix_path = entry["mix_path"]
        src1 = entry["src1"]
        src2 = entry["src2"]
        spk1 = entry["spk1"]
        spk2 = entry["spk2"]

        # mixture
        mix, sr = torchaudio.load(mix_path)
        mix = mix.mean(0)

        # optional WHAM noise corruption
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

        # dual model embeddings: [1,2,256]
        with torch.no_grad():
            ed = dual_model(mix)
        e0, e1 = ed.squeeze(0)  # [2,256]

        # teacher embeddings from clean sources
        t1 = get_teacher_emb(teacher_model, src1, device)
        t2 = get_teacher_emb(teacher_model, src2, device)

        # PIT matching
        score_direct = cosine(e0, t1) + cosine(e1, t2)
        score_swap   = cosine(e0, t2) + cosine(e1, t1)

        if score_direct >= score_swap:
            mapped = [(e0, spk1), (e1, spk2)]
        else:
            mapped = [(e0, spk2), (e1, spk1)]

        for e, lab in mapped:
            all_embs.append(e.cpu().numpy())
            all_labels.append(lab)

    return np.vstack(all_embs), np.array(all_labels)


# =====================================================================
# 4) Clustering metrics
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

    try:
        silhouette = silhouette_score(e, labels)
    except Exception:
        silhouette = float("nan")

    return {
        "same_mean_cos": float(same_mean),
        "diff_mean_cos": float(diff_mean),
        "separation": float(separation),
        "cluster_acc": float(_cluster_accuracy(pred, labels)),
        "nmi": float(normalized_mutual_info_score(labels, pred)),
        "ari": float(adjusted_rand_score(labels, pred)),
        "silhouette": float(silhouette),
    }


# =====================================================================
# 5) JOINT t-SNE VIDEO PIPELINE (fixed speakers + fixed styles + shared t-SNE space)
# =====================================================================
def build_style_map(chosen_speakers):
    chosen = np.array(sorted(chosen_speakers)).astype(int)
    k = len(chosen)
    cmap = plt.cm.get_cmap("turbo", k)
    markers = ['o', '^', 's', 'D', 'v', 'P', 'X', '*', '<', '>', 'h', 'H', 'p', '+', 'x', '1', '2', '3', '4']
    style = {}
    for i, spk in enumerate(chosen):
        style[int(spk)] = {"color": cmap(i), "marker": markers[i % len(markers)]}
    return style

def choose_common_speakers(labels_by_ovlp, overlaps, k=20):
    common = None
    for ovlp in overlaps:
        spk_set = set(np.unique(labels_by_ovlp[ovlp]).tolist())
        common = spk_set if common is None else (common & spk_set)
    common = np.array(sorted(list(common))).astype(int)

    if len(common) == 0:
        raise RuntimeError("No common speakers across overlaps; cannot make consistent TSNE subset.")

    return common[:min(k, len(common))]

def make_joint_tsne_coords(embs_by_ovlp, labels_by_ovlp, overlaps, chosen_speakers,
                           max_per_speaker=40, tsne_perplexity=20, tsne_seed=0):
    rng = np.random.RandomState(tsne_seed)

    X_all = []
    spk_all = []
    ovlp_all = []

    for ovlp in overlaps:
        embs = embs_by_ovlp[ovlp]
        labels = labels_by_ovlp[ovlp]

        for spk in chosen_speakers:
            idx = np.where(labels == spk)[0]
            if len(idx) == 0:
                continue
            if len(idx) > max_per_speaker:
                idx = rng.choice(idx, size=max_per_speaker, replace=False)

            X_all.append(embs[idx])
            spk_all.append(labels[idx])
            ovlp_all.append(np.full(len(idx), ovlp))

    X_all = np.vstack(X_all)
    spk_all = np.concatenate(spk_all).astype(int)
    ovlp_all = np.concatenate(ovlp_all).astype(int)

    # normalize before TSNE (helps if you interpret embeddings via cosine)
    X_all = X_all / (np.linalg.norm(X_all, axis=1, keepdims=True) + 1e-10)

    tsne = TSNE(
        n_components=2,
        perplexity=tsne_perplexity,
        learning_rate="auto",
        init="pca",
        random_state=tsne_seed
    )
    coords_all = tsne.fit_transform(X_all)

    return coords_all, spk_all, ovlp_all

def render_tsne_frames(coords_all, spk_all, ovlp_all, overlaps, chosen_speakers, style_map,
                       frames_dir, dpi=200):
    os.makedirs(frames_dir, exist_ok=True)

    xmin, xmax = coords_all[:, 0].min(), coords_all[:, 0].max()
    ymin, ymax = coords_all[:, 1].min(), coords_all[:, 1].max()

    for ovlp in overlaps:
        mask_ov = (ovlp_all == ovlp)

        plt.figure(figsize=(10, 8))
        for spk in chosen_speakers:
            m = mask_ov & (spk_all == spk)
            if not np.any(m):
                continue
            st = style_map[int(spk)]
            pts = coords_all[m]
            plt.scatter(
                pts[:, 0], pts[:, 1],
                s=22,
                c=[st["color"]],
                marker=st["marker"],
                alpha=0.85,
                linewidths=0.3,
                edgecolors="k"
            )

        plt.title(f"t-SNE (shared space) — Overlap = {ovlp}%")
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.tight_layout()

        out_path = os.path.join(frames_dir, f"frame_{ovlp:03d}.png")
        plt.savefig(out_path, dpi=dpi)
        plt.close()
        print(f"[✔] Saved {out_path}")

    return xmin, xmax, ymin, ymax


# =====================================================================
# 6) Plot metrics vs overlap (separate plots)
# =====================================================================
def save_line_plot(metrics_by_ovlp, ovs, metric_keys, title, ylabel, out_path):
    plt.figure(figsize=(8, 5))
    for k in metric_keys:
        ys = [metrics_by_ovlp[o][k] for o in ovs]
        plt.plot(ovs, ys, marker="o", label=k)
    plt.xlabel("Overlap (%)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"[✔] Saved {out_path}")

def _load_teacher(teacher_ckpt, device):
    teacher = SingleSpkEncoder().to(device)
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


def _load_dual(ckpt_base, ckpt_joint, device):
    dual = load_dual_model(ckpt_base, device=device)
    joint_ckpt = torch.load(ckpt_joint, map_location=device)
    joint_state = joint_trained_model_weights(joint_ckpt['state_dict'])
    dual.load_state_dict(joint_state, strict=True)
    dual.eval()
    return dual


def run_one_overlap_job(args_tuple):
    """
    Runs one overlap on one GPU, saves results to disk, and returns file paths.
    """
    (ovlp, gpu_id, meta_path, ckpt_base, ckpt_joint, teacher_ckpt, out_dir, seed) = args_tuple

    # ----- bind process to GPU -----
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    # ----- make randomness reproducible per overlap -----
    local_seed = seed + int(ovlp) * 1000 + int(gpu_id) * 10
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)
    torch.cuda.manual_seed_all(local_seed)

    # ----- load models on this GPU -----
    teacher = _load_teacher(teacher_ckpt, device)
    dual = _load_dual(ckpt_base, ckpt_joint, device)

    # ----- load metadata -----
    metadata = parse_metadata(meta_path)
    print(f"[GPU {gpu_id}] ovlp={ovlp}: loaded {len(metadata)} mixtures")

    # ----- extract embeddings -----
    embs, labels = extract_dual_embeddings_with_teacher(
        dual_model=dual,
        teacher_model=teacher,
        metadata=metadata,
        device=device,
        verbose=True,
    )

    # ----- compute metrics -----
    res = compute_clustering_metrics(embs, labels)

    # ----- save -----
    npz_path = os.path.join(out_dir, f"ovlp{ovlp:03d}_results.npz")
    json_path = os.path.join(out_dir, f"ovlp{ovlp:03d}_metrics.json")

    np.savez(npz_path, embs=embs, labels=labels)
    with open(json_path, "w") as f:
        json.dump(res, f, indent=2)

    print(f"[GPU {gpu_id}] ovlp={ovlp}: saved {npz_path} and {json_path}")
    return ovlp, npz_path, json_path
# =====================================================================
# 7) MAIN
# =====================================================================
if __name__ == "__main__":
    overlaps = [0, 25, 50, 75, 100]
    NUM_SPEAKERS_TSNE = 20
    MAX_PER_SPK_PER_OVLP = 40

    SEED = 44

    CKPT_BASE = "/mnt/disks/data/model_ckpts/librispeech_asp_ft_wavlm_linear_dualemb_tr360/best-epoch=49-val_separation=0.000.ckpt"
    CKPT_JOINT = "/mnt/disks/data/model_ckpts/pDCCRN_2sp_dpccn_joint_training_freezewavlm_indloss/best-epoch=19-val_separation=0.000.ckpt"
    TEACHER_CKPT = "/mnt/disks/data/model_ckpts/librispeech_asp_wavlm_tr360/best-epoch=62-val_separation=0.000.ckpt"

    out_dir = "/home/sidharth./codebase/wavlm_dual_embedding/analysis/tsne_video_frames_shared"
    os.makedirs(out_dir, exist_ok=True)

    num_gpus = torch.cuda.device_count()
    print("CUDA available:", torch.cuda.is_available(), " num_gpus:", num_gpus)

    # -------------------------------------------------------------
    # (1) Run overlaps in parallel across GPUs
    # -------------------------------------------------------------
    jobs = []
    for i, ovlp in enumerate(overlaps):
        META = f"/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/Libri2Mix_{ovlp}vlp/Libri2Mix_ovl{ovlp}to{ovlp}/wav16k/min/metadata/mixture_test_mix_clean.csv"
        gpu_id = (i % max(1, num_gpus))  # round-robin
        jobs.append((ovlp, gpu_id, META, CKPT_BASE, CKPT_JOINT, TEACHER_CKPT, out_dir, SEED))

    if torch.cuda.is_available() and num_gpus > 1:
        # Use spawn for CUDA safety
        mp.set_start_method("spawn", force=True)
        procs = min(num_gpus, len(jobs))
        print(f"Running {len(jobs)} overlaps using {procs} parallel processes...")
        with mp.Pool(processes=procs) as pool:
            results = pool.map(run_one_overlap_job, jobs)
    else:
        print("Single GPU/CPU detected; running overlaps sequentially...")
        results = [run_one_overlap_job(j) for j in jobs]

    # -------------------------------------------------------------
    # (2) Load saved embeddings/labels + metrics
    # -------------------------------------------------------------
    embs_by_ovlp = {}
    labels_by_ovlp = {}
    metrics_by_ovlp = {}

    for ovlp, npz_path, json_path in results:
        data = np.load(npz_path)
        embs_by_ovlp[ovlp] = data["embs"]
        labels_by_ovlp[ovlp] = data["labels"].astype(int)

        with open(json_path, "r") as f:
            metrics_by_ovlp[ovlp] = json.load(f)

    # -------------------------------------------------------------
    # (3) Shared-space t-SNE frames (fixed speakers + fixed styles)
    # -------------------------------------------------------------
    chosen = choose_common_speakers(labels_by_ovlp, overlaps, k=NUM_SPEAKERS_TSNE)
    print("\nChosen speakers (fixed across overlaps):", chosen.tolist())

    style_map = build_style_map(chosen)

    coords_all, spk_all, ovlp_all = make_joint_tsne_coords(
        embs_by_ovlp, labels_by_ovlp, overlaps, chosen_speakers=chosen,
        max_per_speaker=MAX_PER_SPK_PER_OVLP,
        tsne_perplexity=20,
        tsne_seed=0
    )
    print(f"\nJoint TSNE fitted on {coords_all.shape[0]} points total (shared space).")

    render_tsne_frames(
        coords_all, spk_all, ovlp_all,
        overlaps=overlaps,
        chosen_speakers=chosen,
        style_map=style_map,
        frames_dir=out_dir,
        dpi=220
    )

    # -------------------------------------------------------------
    # (4) Separate metric plots vs overlap
    # -------------------------------------------------------------
    ovs = sorted(metrics_by_ovlp.keys())

    def save_line_plot(metric_keys, title, ylabel, out_name):
        plt.figure(figsize=(8, 5))
        for k in metric_keys:
            ys = [metrics_by_ovlp[o][k] for o in ovs]
            plt.plot(ovs, ys, marker="o", label=k)
        plt.xlabel("Overlap (%)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=9)
        plt.tight_layout()
        out_path = os.path.join(out_dir, out_name)
        plt.savefig(out_path, dpi=220)
        plt.close()
        print(f"[✔] Saved {out_path}")

    save_line_plot(
        ["same_mean_cos", "diff_mean_cos", "separation"],
        "Cosine similarity stats vs overlap",
        "Cosine / separation",
        "metrics_cosine_separation_vs_overlap.png",
    )

    save_line_plot(
        ["cluster_acc", "nmi", "ari"],
        "Clustering quality vs overlap",
        "Score",
        "metrics_clustering_quality_vs_overlap.png",
    )

    save_line_plot(
        ["silhouette"],
        "Silhouette score vs overlap",
        "Silhouette",
        "metrics_silhouette_vs_overlap.png",
    )

    print("\nDone.")
    print("To make a video from TSNE frames:")
    print("  cd", out_dir)
    print("  ffmpeg -framerate 1 -i frame_%03d.png -pix_fmt yuv420p tsne_overlap.mp4")