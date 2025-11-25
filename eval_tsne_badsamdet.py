
import os
import json
import random
import torch
import torchaudio
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from collections import defaultdict

# your dual-speaker model
from model import SpeakerEncoderDualWrapper

random.seed(44)

# -----------------------------------------------------------
# 1. KNN-BASED BAD SAMPLE DETECTION
# -----------------------------------------------------------
def extract_mix_id_from_path(mixpath):
    """
    mixpath example:
    /.../460-172359-0099_6385-220959-0018.wav
    returns mix_id without extension.
    """
    base = os.path.basename(mixpath)
    return base.replace(".wav", "")
def load_snr_table(csv_path="/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/Libri2Mix/wav16k/min/metadata/metrics_train-100_mix_clean.csv"):
    """
    Reads SNR table:
    mixture_ID, source_1_SNR, source_2_SNR
    Returns dict: { mixture_ID : (snr1, snr2) }
    """
    df = pd.read_csv(csv_path)
    snr_map = {}
    for _, row in df.iterrows():
        mix_id = str(row['mixture_ID']).strip()
        snr_map[mix_id] = (float(row['source_1_SNR']),
                           float(row['source_2_SNR']))
    return snr_map


def compute_snr_stats_for_failures(
    bad_indices,
    mixpaths,
    snr_map
):
    """
    bad_indices: list of embedding indices (each embedding corresponds to a mix)
    mixpaths: array of mixpaths aligned w/ embedding indices
    snr_map: { mix_id : (snr1, snr2) }
    """

    fail_snr1 = []
    fail_snr2 = []
    succ_snr1 = []
    succ_snr2 = []

    N = len(mixpaths)
    bad_set = set(int(i) for i in bad_indices)

    for i in range(N):
        mixpath = mixpaths[i]
        mix_id = extract_mix_id_from_path(mixpath)

        if mix_id not in snr_map:
            continue

        snr1, snr2 = snr_map[mix_id]

        if i in bad_set:
            fail_snr1.append(snr1)
            fail_snr2.append(snr2)
        else:
            succ_snr1.append(snr1)
            succ_snr2.append(snr2)

    def mean_std(vals):
        if len(vals) == 0:
            return (None, None)
        return (float(np.mean(vals)), float(np.std(vals)))

    return {
        "fail": {
            "source1": mean_std(fail_snr1),
            "source2": mean_std(fail_snr2),
            "count": len(fail_snr1),
        },
        "success": {
            "source1": mean_std(succ_snr1),
            "source2": mean_std(succ_snr2),
            "count": len(succ_snr1),
        }
    }

def detect_bad_samples_knn(embs, labels, K=20, threshold=0.30):
    """
    For each embedding:
      mixing_ratio = (# K nearest neighbors with DIFFERENT speaker) / K

    If mixing_ratio > threshold → bad sample
    """
    N = embs.shape[0]

    nn = NearestNeighbors(n_neighbors=K+1, metric="euclidean")
    nn.fit(embs)

    distances, neighbors = nn.kneighbors(embs)  # neighbors[i][0] is itself

    mixing_ratios = np.zeros(N)
    bad_indices = []

    for i in range(N):
        nbr_ids = neighbors[i][1:]        # remove itself
        nbr_lbls = labels[nbr_ids]

        same = (nbr_lbls == labels[i])
        diff = (~same)

        mixing_ratios[i] = diff.sum() / K

        if mixing_ratios[i] > threshold:
            bad_indices.append(i)

    return np.array(bad_indices), mixing_ratios


def build_knn_bad_records(bad_idx, labels, mixpaths, mixing_ratios):
    records = []
    for i in bad_idx:
        records.append({
            "index": int(i),
            "mixpath": str(mixpaths[i]),
            "speaker": int(labels[i]),
            "query_idx": int(i % 2),
            "mixing_ratio": float(mixing_ratios[i])
        })
    return records


# -----------------------------------------------------------
# 2. Model utilities
# -----------------------------------------------------------
def strip_dual_model_weights(state):
    new_state = {}
    for k, v in state.items():

        if not k.startswith("model."):
            continue

        k2 = k[len("model."):]

        # remove unrelated modules
        if k2.startswith("single_sp_model."):
            continue
        if k2.startswith("arcface_loss."):
            continue

        new_state[k2] = v
    return new_state


def load_dual_model(path, emb_dim=256, device="cuda"):
    ckpt = torch.load(path, map_location=device)
    state = strip_dual_model_weights(ckpt["state_dict"])

    model = SpeakerEncoderDualWrapper(emb_dim=emb_dim)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


# -----------------------------------------------------------
# 3. LibriMix metadata parser
# -----------------------------------------------------------
def parse_metadata(csv_path):
    meta = []
    with open(csv_path, "r") as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 7:
                continue
            mix_id, mix_path, src1, src2, spk1, spk2, length = parts
            if spk1 == "speaker_1_ID":
                continue
            meta.append({
                "mix_path": mix_path,
                "src1": src1,
                "src2": src2,
                "spk1": int(spk1),
                "spk2": int(spk2)
            })
    return meta


# -----------------------------------------------------------
# 4. Gender map
# -----------------------------------------------------------
def parse_speaker_gender(path):
    mapping = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 2:
                continue
            try:
                spk_id = int(parts[0])
            except:
                continue
            mapping[spk_id] = parts[1]
    return mapping


# -----------------------------------------------------------
# 5. Choose speakers
# -----------------------------------------------------------
def select_speakers(metadata, min_mixtures=20, num_speakers=4):
    speaker_to_files = {}

    for entry in metadata:
        m = entry["mix_path"]
        s1, s2 = entry["spk1"], entry["spk2"]

        speaker_to_files.setdefault(s1, []).append(m)
        speaker_to_files.setdefault(s2, []).append(m)

    valid = {s: m for s, m in speaker_to_files.items() if len(m) >= min_mixtures}
    chosen = random.sample(list(valid.keys()), num_speakers)

    return chosen, valid


# -----------------------------------------------------------
# 6. Extract embeddings
# -----------------------------------------------------------
def extract_dual_embeddings(model, speaker_files, gender_map, max_per_spk=40, device="cuda"):
    all_embs = []
    all_speakers = []
    all_mixpaths = []
 
    wav, sr = torchaudio.load('/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/Libri2_0.5ovl/Libri2Mix/wav16k/min/dev/mix_clean/84-121123-0001_5895-34629-0002.wav')
    wav = wav.mean(0)
    wav = wav.to(device).unsqueeze(0)
    with torch.no_grad():
        emb, Q, attn_weights = model(wav, return_queries=True)
    
    import matplotlib.pyplot as plt

    A = attn_weights.squeeze(0)   # [4, 2, 366]
    H, Q, T = A.shape

    for h in range(H):
        for q in range(Q):
            plt.figure(figsize=(12, 2))
            plt.plot(A[h, q].cpu().numpy())
            plt.title(f"Attention — Head (50%ovlp audio) {h}, Query {q}")
            plt.xlabel("Time frames")
            plt.ylabel("Weight")
            plt.ylim(0, A[h, q].max()*1.1)
            plt.savefig(f'attn_Weights_For_0.5ovlp_H_unnormalized{h}_q{q}.png')
    breakpoint()
    for spk, mix_list in speaker_files.items():

        chosen = random.sample(mix_list, min(len(mix_list), max_per_spk))

        for mix_path in chosen:
            wav, sr = torchaudio.load(mix_path)
            wav = wav.mean(0)
            wav = wav.to(device).unsqueeze(0)

            with torch.no_grad():
                emb, _ = model(wav, return_queries=True)  # [1,2,D]

            emb = emb.squeeze(0).cpu().numpy()

            all_embs.append(emb[0]); all_speakers.append(spk); all_mixpaths.append(mix_path)
            all_embs.append(emb[1]); all_speakers.append(spk); all_mixpaths.append(mix_path)

    return np.vstack(all_embs), np.array(all_speakers), np.array(all_mixpaths)


# -----------------------------------------------------------
# 7. Gender contrast (your old function unchanged)
# -----------------------------------------------------------
def compute_gender_contrast_per_speaker(labels, mixpaths, metadata, gmap):

    mix_lookup = {m["mix_path"]: (m["spk1"], m["spk2"]) for m in metadata}

    stats = {}
    for spk in np.unique(labels):
        stats[spk] = {"same_gender": 0, "diff_gender": 0, "ratio_diff_gender": 0.0}

    for lbl, mix in zip(labels, mixpaths):
        if mix not in mix_lookup:
            continue

        s1, s2 = mix_lookup[mix]
        other = s2 if lbl == s1 else s1

        g1 = gmap.get(lbl, "U")
        g2 = gmap.get(other, "U")

        if g1 == g2:
            stats[lbl]["same_gender"] += 1
        else:
            stats[lbl]["diff_gender"] += 1

    for spk in stats:
        tot = stats[spk]["same_gender"] + stats[spk]["diff_gender"]
        if tot > 0:
            stats[spk]["ratio_diff_gender"] = stats[spk]["diff_gender"] / tot

    return stats


# -----------------------------------------------------------
# 8. t-SNE plotting
# -----------------------------------------------------------
def plot_tsne(coords, speakers, contrast_stats, bad_indices, save_path):

    plt.figure(figsize=(11,9))
    uniq = np.unique(speakers)

    for spk in uniq:
        idx = (speakers == spk)
        XY = coords[idx]

        plt.scatter(XY[:,0], XY[:,1], s=18, alpha=0.7, label=f"Spk {spk}")

        cx, cy = XY[:,0].mean(), XY[:,1].mean()

        d = contrast_stats[spk]

        txt = f"Spk {spk}\nDiff: {d['diff_gender']}\nSame: {d['same_gender']}\nRatio:{d['ratio_diff_gender']:.2f}"

        plt.text(cx, cy, txt, fontsize=9,
                 bbox=dict(facecolor="white", alpha=0.7, edgecolor="black"))

    # highlight bad samples
    if len(bad_indices) > 0:
        bx = coords[bad_indices,0]
        by = coords[bad_indices,1]
        plt.scatter(bx, by,
                    s=80,
                    facecolors="none",
                    edgecolors="red",
                    linewidths=2,
                    label="Bad Samples")

    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print("Saved:", save_path)


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
if __name__ == "__main__":

    META = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/Libri2Mix/wav16k/min/metadata/mixture_train-100_mix_clean.csv"
    CKPT = "/mnt/disks/data/model_ckpts/librispeech_asp_wavlm_ft_dualemb_queryorthogonality_mhqa/best-epoch=54-val_separation=0.000.ckpt"
    SPEAKER_TXT = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/LibriSpeech/SPEAKERS.TXT"

    SAVE_TSNE = "/home/sidharth./codebase/wavlm_dual_embedding/analysis/tsne/tsne_dual_train_seed44_queryorthogonal_mhqa_ft_with_gendercontrast_1_overlap_badsample_detection.png"
    SAVE_JSON = "/home/sidharth./codebase/wavlm_dual_embedding/analysis/tsne/tsne_dual_train_seed44_queryorthogonal_mhqa_ft_with_gendercontrast_1_overlap_bad_samples_knn.json"

    metadata = parse_metadata(META)
    gmap = parse_speaker_gender(SPEAKER_TXT)
    model = load_dual_model(CKPT, device="cpu")

    chosen, valid_map = select_speakers(metadata, min_mixtures=20, num_speakers=4)
    print("Chosen speakers:", chosen)

    speaker_files = {s: valid_map[s] for s in chosen}

    embs, labels, mixpaths = extract_dual_embeddings(model, speaker_files, gmap, device="cpu")

    coords = TSNE(perplexity=20, learning_rate="auto", init="pca", random_state=0).fit_transform(embs)

    # -------- KNN detection ----------
    bad_knn, mixing_ratios = detect_bad_samples_knn(embs, labels, K=20, threshold=0.30)
    print(f"[✔] KNN outliers: {len(bad_knn)}")

    bad_records = build_knn_bad_records(bad_knn, labels, mixpaths, mixing_ratios)

    with open(SAVE_JSON, "w") as f:
        json.dump(bad_records, f, indent=2)
    print("Saved JSON:", SAVE_JSON)

    contrast = compute_gender_contrast_per_speaker(labels, mixpaths, metadata, gmap)

    plot_tsne(coords, labels, contrast, bad_knn, SAVE_TSNE)
    SNR_CSV = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/Libri2Mix/wav16k/min/metadata/metrics_train-100_mix_clean.csv"
    snr_map = load_snr_table(SNR_CSV)

    # ----- Compute SNR stats -----
# ----- Load SNR CSV -----
    SNR_CSV = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/Libri2Mix/wav16k/min/metadata/metrics_train-100_mix_clean.csv"
    snr_map = load_snr_table(SNR_CSV)

    # ----- Compute SNR stats -----
    snr_stats = compute_snr_stats_for_failures(
        bad_knn,    # <<< FIXED
        mixpaths,
        snr_map
    )

    print("\n========== SNR STATISTICS ==========")
    print(f"Failure cases ({snr_stats['fail']['count']} samples):")
    print(f"  Source 1 SNR mean/std: {snr_stats['fail']['source1']}")
    print(f"  Source 2 SNR mean/std: {snr_stats['fail']['source2']}")

    print(f"\nSuccess cases ({snr_stats['success']['count']} samples):")
    print(f"  Source 1 SNR mean/std: {snr_stats['success']['source1']}")
    print(f"  Source 2 SNR mean/std: {snr_stats['success']['source2']}")
    print("====================================\n")

