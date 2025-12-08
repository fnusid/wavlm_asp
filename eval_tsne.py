import os
import json
import random
import torch
import torchaudio
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

from model import SpeakerEncoderWrapper    # your model
from dataset import Librispeech            # your dataset class

random.seed(42)

# -------------------------------------------------
# 1. Load the model checkpoint
# -------------------------------------------------
def load_model(ckpt_path, emb_dim=256, device="cuda"):
    state = torch.load(ckpt_path, map_location=device)
    model = SpeakerEncoderWrapper(emb_dim=emb_dim)

    model.load_state_dict(state["state_dict"], strict=False)
    model.to(device)
    model.eval()
    return model


# -------------------------------------------------
# 2. Select speakers with minimum number of samples
# -------------------------------------------------
def select_random_speakers(metadata, min_samples=20, num_speakers=4):
    """
    metadata: list of file paths
    returns: list of selected speaker IDs
    """
    speaker_to_files = {}

    for path in metadata:
        spk = int(path.split("/")[-3])
        speaker_to_files.setdefault(spk, []).append(path)

    # Filter speakers with enough samples
    valid = {s: f for s, f in speaker_to_files.items() 
             if len(f) >= min_samples}

    if len(valid) < num_speakers:
        raise ValueError(
            f"Not enough speakers with >= {min_samples} samples."
        )

    return random.sample(list(valid.keys()), num_speakers), valid


# -------------------------------------------------
# 3. Extract embeddings for each selected speaker
# -------------------------------------------------
def extract_embeddings(model, speaker_files, max_per_speaker=40, device="cuda"):
    all_embs = []
    all_labels = []

    for speaker_id, file_list in speaker_files.items():
        # pick limited number per speaker
        files = random.sample(file_list, min(len(file_list), max_per_speaker))

        for wav_path in files:
            wav, sr = torchaudio.load(wav_path)
            wav = wav.mean(dim=0)         # mono
            wav = wav.to(device).unsqueeze(0)

            with torch.no_grad():
                emb = model(wav)          # [1, emb_dim]

            all_embs.append(emb.cpu().numpy())
            all_labels.append(speaker_id)

    all_embs = np.vstack(all_embs)
    all_labels = np.array(all_labels)
    return all_embs, all_labels


# -------------------------------------------------
# 4. Plot t-SNE
# -------------------------------------------------
def plot_tsne(embeddings, labels, save_path):
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        max_iter=1500,
        init="pca",
        random_state=0
    )
    points = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    speakers = np.unique(labels)

    for spk in speakers:
        idx = labels == spk
        plt.scatter(points[idx, 0], points[idx, 1], label=str(spk), alpha=0.7)

    plt.legend()
    plt.title("t-SNE of Speaker Embeddings")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved t-SNE to {save_path}")


# -------------------------------------------------
# 5. Main
# -------------------------------------------------
if __name__ == "__main__":

    # --- paths ---
    METADATA = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/LibriSpeech/libri_dev_clean.txt"
    # CKPT = "/mnt/disks/data/model_ckpts/librispeech_asp_wavlm/best-epoch=47-val_separation=0.000.ckpt"
    CKPT = "/mnt/disks/data/model_ckpts/librispeech_asp_wavlm_tr360/best-epoch=62-val_separation=0.000.ckpt"
    SAVE_FIG = "/home/sidharth./codebase/wavlm_single_embedding/analysis/tsne/tsne_selected_DEV_single_speakers_tr360.png"
    
    # --- load metadata ---
    with open(METADATA, "r") as f:
        metadata = [x.strip() for x in f if x.strip()]

    # --- load model ---
    model = load_model(CKPT, emb_dim=256)

    # --- choose speakers ---
    selected_speakers, valid_map = select_random_speakers(
        metadata, 
        min_samples=20, 
        num_speakers=4
    )

    print(f"Selected speakers: {selected_speakers}")

    speaker_files = {s: valid_map[s] for s in selected_speakers}

    # --- extract embeddings ---
    embs, labels = extract_embeddings(model, speaker_files)

    # --- plot ---
    plot_tsne(embs, labels, SAVE_FIG)
