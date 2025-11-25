import os
import json
import random
import torch
import torchaudio
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

from model import SpeakerEncoderDualWrapper   # your dual model

random.seed(44)

def strip_dual_model_weights(state):
    """
    Extract ONLY the dual-speaker model weights from a Lightning checkpoint.

    Keeps keys like:
        model.wavlm.*
        model.encoder1.*
        model.encoder2.*
        model.projector.*

    Removes:
        arcface_loss.*
        single_sp_model.*
        anything outside model.*
    """
    new_state = {}

    for k, v in state.items():

        # 1) Keep only dual model parameters
        if not k.startswith("model."):
            continue

        # 2) Remove 'model.' prefix
        clean_k = k[len("model."):]

        # 3) Skip teacher or classification head if any appear
        if clean_k.startswith("single_sp_model."):
            continue
        if clean_k.startswith("arcface_loss."):
            continue

        # 4) Add only clean dual model params
        new_state[clean_k] = v

    return new_state


# -------------------------------------------------
# 1. Load Dual Model
# -------------------------------------------------
def load_dual_model(ckpt_path, emb_dim=256, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"]
    state = strip_dual_model_weights(state)
    model = SpeakerEncoderDualWrapper(emb_dim=emb_dim)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


# -------------------------------------------------
# 2. Parse CSV Metadata (your 7-field format)
# -------------------------------------------------
def parse_metadata(csv_path):
    """
    Parses CSV with 7 fields and skips the header.
    """
    metadata = []
    with open(csv_path, "r") as f:
        header = next(f)  # skip the first line

        for line in f:
            parts = line.strip().split(",")

            if len(parts) != 7:
                continue

            mix_id, mix_path, src1, src2, spk1, spk2, length = parts

            # guard against accidental header lines elsewhere
            if spk1 == "speaker_1_ID":
                continue

            metadata.append({
                "mix_path": mix_path,
                "src1": src1,
                "src2": src2,
                "spk1": int(spk1),
                "spk2": int(spk2)
            })

    return metadata


# -------------------------------------------------
# 3. Select speakers with ≥ min mixtures
# -------------------------------------------------
def select_speakers(metadata, min_mixtures=20, num_speakers=4):

    speaker_to_mixfiles = {}

    for entry in metadata:
        mix = entry["mix_path"]
        s1 = entry["spk1"]
        s2 = entry["spk2"]

        speaker_to_mixfiles.setdefault(s1, []).append(mix)
        speaker_to_mixfiles.setdefault(s2, []).append(mix)

    # filter speakers
    valid = {s: m for s, m in speaker_to_mixfiles.items()
             if len(m) >= min_mixtures}

    if len(valid) < num_speakers:
        raise ValueError(
            f"Only {len(valid)} speakers have ≥{min_mixtures} mixtures."
        )

    chosen = random.sample(list(valid.keys()), num_speakers)
    return chosen, valid


# -------------------------------------------------
# 4. Extract embeddings from model (dual)
# -------------------------------------------------
def extract_dual_embeddings(model, speaker_files, max_per_spk=40, device="cuda"):

    all_embs = []
    all_lbls = []

    for spk, mix_list in speaker_files.items():

        chosen_mixes = random.sample(
            mix_list,
            min(len(mix_list), max_per_spk)
        )

        for mix_path in chosen_mixes:

            wav, sr = torchaudio.load(mix_path)
            wav = wav.mean(0)                 # mono
            wav = wav.to(device).unsqueeze(0) # [1,T]

            with torch.no_grad():
                emb = model(wav)              # [1,2,256]

            emb = emb.squeeze(0).cpu().numpy()  # [2,256]

            # append both embeddings with same speaker label
            all_embs.append(emb[0])
            all_lbls.append(spk)

            all_embs.append(emb[1])
            all_lbls.append(spk)

    return np.vstack(all_embs), np.array(all_lbls)


# -------------------------------------------------
# 5. t-SNE Plot
# -------------------------------------------------
def plot_tsne(embs, labels, save_path):
    #based on labels, get all the embs corresponding to each unique label, take mean and calculate the cosine similarity for each pair of centroids
    unique_speakers = np.unique(labels)
    centroids = []
    for spk in unique_speakers:
        spk_embs = embs[labels == spk]
        centroid = np.mean(spk_embs, axis=0)

        centroids.append(centroid)

    centroids = np.vstack(centroids)
    print("Centroid cosine similarity matrix:")
    centroids_norm = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    cos_sim_matrix = np.dot(centroids_norm, centroids_norm.T)
    print(cos_sim_matrix)
    
    
    tsne = TSNE(
        n_components=2,
        perplexity=20,
        learning_rate="auto",
        init="pca",
        random_state=0
    )

    coords = tsne.fit_transform(embs)

    plt.figure(figsize=(10,8))
    for spk in np.unique(labels):
        idx = labels == spk
        plt.scatter(coords[idx,0], coords[idx,1], label=f"Spk {spk}", s=18)

    plt.legend()
    plt.title("t-SNE of Dual Speaker Embeddings")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[✔] Saved t-SNE to {save_path}")


# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":

    # META = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/Libri2Mix/wav16k/min/metadata/mixture_dev_mix_clean.csv"
    META="/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/Libri2Mix/wav16k/min/metadata/mixture_train-100_mix_clean.csv"
    CKPT = "/mnt/disks/data/model_ckpts/librispeech_asp_wavlm_ft_dualemb_orthogonality_mhqa/best-epoch=29-val_separation=0.000.ckpt"
    SAVE = "/home/sidharth./codebase/wavlm_dual_embedding/analysis/tsne/tsne_dual_train_seed44_orthogonal_mhqa_ft.png"

    # Load metadata
    metadata = parse_metadata(META)

    # Load model
    model = load_dual_model(CKPT)

    # Select speakers with enough occurrences
    chosen, valid_map = select_speakers(
        metadata, 
        min_mixtures=20,
        num_speakers=4
    )
    print("Selected speakers:", chosen)

    speaker_files = {s: valid_map[s] for s in chosen}

    # Extract embeddings
    embs, labels = extract_dual_embeddings(
        model, speaker_files
    )

    # Plot
    plot_tsne(embs, labels, SAVE)
