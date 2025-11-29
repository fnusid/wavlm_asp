import os
import json
import torch
import torchaudio
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
from sklearn.manifold import TSNE

from model import SpeakerEncoderDualWrapper

sys.path.append("/home/sidharth./codebase/")

from wavlm_single_embedding.model import SpeakerEncoderWrapper as SingleSpeakerEncoderWrapper
# ============================================================
# Helpers
# ============================================================

def extract_mix_id_from_path(path):
    return os.path.basename(path).replace(".wav", "")

def load_snr_csv(path):
    df = pd.read_csv(path)
    snr_map = {}
    for _, row in df.iterrows():
        snr_map[row["mixture_ID"]] = (
            float(row["source_1_SNR"]),
            float(row["source_2_SNR"]),
        )
    return snr_map

# ============================================================
# KNN Mixing / Purity / Confusion
# ============================================================

def compute_knn_metrics(embs, labels, K=20):
    N = len(embs)
    nbrs = NearestNeighbors(n_neighbors=K+1).fit(embs)
    distances, neighbors = nbrs.kneighbors(embs)

    mixing_ratio = np.zeros(N)
    purity = np.zeros(N)
    speakers = np.unique(labels)
    confusion = {a: {b: 0 for b in speakers} for a in speakers}

    for i in range(N):
        nbr_ids = neighbors[i][1:]
        nbr_lbls = labels[nbr_ids]

        same = np.sum(nbr_lbls == labels[i])
        diff = len(nbr_lbls) - same

        purity[i] = same / K
        mixing_ratio[i] = diff / K

        for lbl in nbr_lbls:
            confusion[labels[i]][lbl] += 1

    return mixing_ratio, purity, confusion

def normalize_confusion(conf):
    norm = {}
    for spk, row in conf.items():
        s = sum(row.values())
        if s == 0:
            norm[spk] = {k: 0.0 for k in row}
        else:
            norm[spk] = {k: row[k] / s for k in row}
    return norm

# ============================================================
# SNR Stats
# ============================================================

def compute_snr_stats(bad_idx, mixpaths, snr_map):
    fail1, fail2 = [], []
    succ1, succ2 = [], []

    bad = set(bad_idx)

    for i, path in enumerate(mixpaths):
        mid = extract_mix_id_from_path(path)
        if mid not in snr_map:
            continue

        s1, s2 = snr_map[mid]
        if i in bad:
            fail1.append(s1)
            fail2.append(s2)
        else:
            succ1.append(s1)
            succ2.append(s2)

    def ms(x):
        return (float(np.mean(x)), float(np.std(x))) if x else (None, None)

    return {
        "fail": {
            "count": len(fail1),
            "source1": ms(fail1),
            "source2": ms(fail2),
        },
        "success": {
            "count": len(succ1),
            "source1": ms(succ1),
            "source2": ms(succ2),
        }
    }

# ============================================================
# Failure Rate per Speaker
# ============================================================

def compute_failure_rate(labels, bad_idx):
    speakers = np.unique(labels)
    bset = set(bad_idx)
    out = {}

    for spk in speakers:
        all_i = np.where(labels == spk)[0]
        fails = [i for i in all_i if i in bset]

        out[int(spk)] = {
            "total": len(all_i),
            "fails": len(fails),
            "fail_rate": len(fails) / len(all_i)
        }

    return out

# ============================================================
# Loading LibriMix Metadata & Model
# ============================================================

def parse_metadata(csv):
    meta = []
    with open(csv, "r") as f:
        next(f)
        for ln in f:
            m_id, path, s1p, s2p, sp1, sp2, _ = ln.strip().split(",")
            meta.append({
                "mix_path": path,
                "spk1": int(sp1),
                "spk2": int(sp2),
            })
    return meta

def load_speaker_gender(txt):
    g = {}
    with open(txt, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith(";"):
                continue
            spk, gender = ln.split("|")[:2]
            g[int(spk.strip())] = gender.strip()
    return g

def strip_dual_weights(state):
    new = {}
    for k,v in state.items():
        if not k.startswith("model."):
            continue
        k2 = k.replace("model.", "")
        if k2.startswith("single_sp_model.") or k2.startswith("arcface_loss."):
            continue
        new[k2] = v
    return new

def load_dual_model(ckpt, device="cpu", emb_dim=256):
    ck = torch.load(ckpt, map_location=device)
    st = strip_dual_weights(ck["state_dict"])
    model = SpeakerEncoderDualWrapper(emb_dim=emb_dim)
    model.load_state_dict(st, strict=True)
    model.to(device).eval()
    return model

# ============================================================
# Extract Embeddings
# ============================================================

def extract_embeddings(model, chosen_spk_files, max_per_spk=40, device="cpu"):
    embs, lbls, paths = [], [], []

    for spk, files in chosen_spk_files.items():
        pick = files[:max_per_spk]

        for path in pick:
            wav, sr = torchaudio.load(path)
            wav = wav.mean(0).to(device).unsqueeze(0)

            with torch.no_grad():
                out, _ = model(wav, return_queries=True)

            e = out.squeeze(0).cpu().numpy()  # [2,256]
            embs.append(e[0]); lbls.append(spk); paths.append(path)
            embs.append(e[1]); lbls.append(spk); paths.append(path)

    return np.vstack(embs), np.array(lbls), np.array(paths)
def convert_keys_to_int(d):
    out = {}
    for k, v in d.items():
        k2 = int(k)  # convert np.int64 → int
        if isinstance(v, dict):
            out[k2] = convert_keys_to_int(v)
        else:
            out[k2] = v
    return out
# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    META = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/Libri2Mix/wav16k/min/metadata/mixture_dev_mix_clean.csv"
    CKPT = "/mnt/disks/data/model_ckpts/librispeech_asp_wavlm_ft_dualemb_queryorthogonality_mhqa/best-epoch=54-val_separation=0.000.ckpt"
    SPEAKER_TXT = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/LibriSpeech/SPEAKERS.TXT"
    SNR_CSV = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/Libri2Mix/wav16k/min/metadata/metrics_dev_mix_clean.csv"

    metadata = parse_metadata(META)
    gmap = load_speaker_gender(SPEAKER_TXT)

    # group files per speaker
    spk_files = defaultdict(list)
    for m in metadata:
        spk_files[m["spk1"]].append(m["mix_path"])
        spk_files[m["spk2"]].append(m["mix_path"])

    # pick 4 random speakers
    speakers = sorted([s for s in spk_files if len(spk_files[s]) >= 20])
    chosen = speakers[:4]
    chosen_files = {s: spk_files[s] for s in chosen}

    print("Selected speakers:", chosen)

    model = load_dual_model(CKPT, device="cpu")
    embs, labels, mixpaths = extract_embeddings(model, chosen_files, device="cpu")

    # --- Compute diagnostics ---
    mixing_ratio, purity, confusion = compute_knn_metrics(embs, labels, K=20)
    bad_idx = np.where(mixing_ratio > 0.30)[0]

    snr_map = load_snr_csv(SNR_CSV)
    snr_stats = compute_snr_stats(bad_idx, mixpaths, snr_map)
    fail_rate = compute_failure_rate(labels, bad_idx)
    confusion_norm = normalize_confusion(confusion)

    # Print SUMMARY
    print("\n=== SUMMARY ===")
    print("Total embeddings:", len(embs))
    print("Bad embeddings:", len(bad_idx))
    print("Overall mixing ratio mean/std:", np.mean(mixing_ratio), np.std(mixing_ratio))
    print("Overall purity mean/std:", np.mean(purity), np.std(purity))
    print("\nSNR Stats:", json.dumps(snr_stats, indent=2))
    print("\nFailure Rate:", json.dumps(fail_rate, indent=2))
    confusion_norm_clean = convert_keys_to_int(confusion_norm)

    print("\nNorm Confusion Matrix:", json.dumps(confusion_norm_clean, indent=2))

