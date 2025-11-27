import os
import json
import random
import torch
import torchaudio
import torch.nn 
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

from model import SpeakerEncoderDualWrapper   # your dual model

random.seed(44)
noise_dir = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/wham_noise/tt"  
noise_files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith(".wav")]
add_noise=False

def mix_with_snr(clean, noise, snr_db):
    """
    clean, noise: torch tensors [T]
    snr_db: desired SNR = clean_power / noise_power (in dB)
    """
    # match lengths
   
    if noise.ndim > 1:
        noise = noise.mean(0)
    if len(noise) < len(clean):
        diff = len(clean) - len(noise)
        padded_noise = F.pad(noise, (diff//2, diff - (diff//2)))
        noise = padded_noise
    else:
        noise = noise[:len(clean)]

    # compute powers
    clean_power = clean.pow(2).mean()
    noise_power  = noise.pow(2).mean()

    # scaling for noise to achieve target SNR
    target_noise_power = clean_power / (10 ** (snr_db / 10))
    scale = torch.sqrt(target_noise_power / (noise_power + 1e-8))

    return clean + scale * noise

# =====================================================================
# 1) Clean dual-model weight loading
# =====================================================================

def strip_dual_model_weights(state):
    new_state = {}
    for k, v in state.items():
        if not k.startswith("model."):
            continue

        clean_k = k[len("model."):]

        if clean_k.startswith("single_sp_model."):
            continue
        if clean_k.startswith("arcface_loss."):
            continue

        new_state[clean_k] = v

    return new_state


def load_dual_model(ckpt_path, emb_dim=256, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"]
    state = strip_dual_model_weights(state)

    model = SpeakerEncoderDualWrapper(emb_dim=emb_dim)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    return model


# =====================================================================
# 2) LibriMix metadata parser
# =====================================================================

def parse_metadata(csv_path):
    metadata = []

    with open(csv_path, "r") as f:
        header = next(f)

        for line in f:
            parts = line.strip().split(",")
            if csv_path.split('/')[-1].split('_')[-1] == 'both.csv':
                if len(parts) != 8: #add noise in header
                    continue
                else:
                    mix_id, mix_path, src1, src2, spk1, spk2, noise, length = parts #mix id, mix_path,source_1_path,speaker_1_ID,source_2_path,speaker_2_ID,noise_path,length

            else:
                if len(parts) != 7:
                    continue
                else:


                    mix_id, mix_path, src1, src2, spk1, spk2, length = parts #mixture_ID,mixture_path,source_1_path,source_2_path,speaker_1_ID,speaker_2_ID,length

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


# =====================================================================
# 3) Parse speakers.txt for gender map
# =====================================================================

def parse_speaker_gender(speaker_txt_path):
    """
    Returns: { speaker_id:int -> gender:'M'/'F' }
    """
    mapping = {}
    with open(speaker_txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 2:
                continue

            spk_id, gender = parts[0], parts[1]

            try:
                spk_id = int(spk_id)
            except:
                continue

            mapping[spk_id] = gender

    return mapping


# =====================================================================
# 4) Choose speakers with enough mixtures
# =====================================================================

def select_speakers(metadata, min_mixtures=20, num_speakers=4):

    speaker_to_mixfiles = {}
    
    for entry in metadata:
        mix = entry["mix_path"]
        s1, s2 = entry["spk1"], entry["spk2"]

        speaker_to_mixfiles.setdefault(s1, []).append(mix)
        speaker_to_mixfiles.setdefault(s2, []).append(mix)
    
    valid = {s: m for s, m in speaker_to_mixfiles.items() if len(m) >= min_mixtures}

    if len(valid) < num_speakers:
        raise ValueError(
            f"Only {len(valid)} speakers have ≥{min_mixtures} mixtures."
        )

    chosen = random.sample(list(valid.keys()), num_speakers)
    return chosen, valid


# =====================================================================
# 5) Extract dual embeddings AND mixture paths
# =====================================================================

def extract_dual_embeddings(model, speaker_files, gender_map, max_per_spk=40, device="cuda"):

    all_embs = []
    all_speakers = []
    all_mixpaths = []

    for spk, mix_list in speaker_files.items():

        chosen_mixes = random.sample(
            mix_list,
            min(len(mix_list), max_per_spk)
        )

        for mix_path in chosen_mixes:

            wav, sr = torchaudio.load(mix_path)
            wav = wav.mean(0)
            if add_noise:
                noise_p = random.choice(noise_files)
                noise_wav, n_sr = torchaudio.load(noise_p)
                r = random.random()
                if r < 0.4:
                    snr = random.uniform(-5, 5)
                elif r < 0.8:
                    snr = random.uniform(5, 15)
                else:
                    snr = random.uniform(15, 25)
                wav = mix_with_snr(wav, noise_wav, snr)
              
            wav = wav.to(device).unsqueeze(0)

            with torch.no_grad():
                emb, _ = model(wav, return_queries=True)   # [1,2,D]

            emb = emb.squeeze(0).cpu().numpy()  # [2,D]

            # each mix contributes 2 embeddings (2 queries)
            all_embs.append(emb[0]); all_speakers.append(spk); all_mixpaths.append(mix_path)
            all_embs.append(emb[1]); all_speakers.append(spk); all_mixpaths.append(mix_path)

    return (
        np.vstack(all_embs),
        np.array(all_speakers),
        np.array(all_mixpaths)
    )


# =====================================================================
# 6) Compute gender contrast (ratio of cross-gender mixtures)
# =====================================================================

def compute_gender_contrast_per_speaker(labels, mixpaths, metadata, gender_map):
    """
    labels: [N] speaker IDs
    mixpaths: [N] mix paths for each embedding
    metadata: parsed metadata
    gender_map: speaker_id -> 'M'/'F'
    """
    # Build lookup: mix_path -> (spk1, spk2)
    mix_lookup = {}
    for entry in metadata:
        mix_lookup[entry['mix_path']] = (entry['spk1'], entry['spk2'])

    speakers = np.unique(labels)
    stats = {
        spk: {"same_gender": 0, "diff_gender": 0, "ratio_diff_gender": 0.0}
        for spk in speakers
    }

    for lbl, mix in zip(labels, mixpaths):
        if mix not in mix_lookup:
            continue

        s1, s2 = mix_lookup[mix]
        co = s2 if lbl == s1 else s1

        g_lbl = gender_map.get(lbl, "U")
        g_co  = gender_map.get(co,  "U")

        if g_lbl == g_co:
            stats[lbl]["same_gender"] += 1
        else:
            stats[lbl]["diff_gender"] += 1

    for spk in speakers:
        total = stats[spk]["same_gender"] + stats[spk]["diff_gender"]
        if total > 0:
            stats[spk]["ratio_diff_gender"] = (
                stats[spk]["diff_gender"] / total
            )

    return stats


# =====================================================================
# 7) t-SNE Plot
# =====================================================================

def plot_tsne(embs, speakers, contrast_stats, save_path):

    tsne = TSNE(
        n_components=2,
        perplexity=20,
        learning_rate="auto",
        init="pca",
        random_state=0
    )

    coords = tsne.fit_transform(embs)

    plt.figure(figsize=(11, 9))

    unique_speakers = np.unique(speakers)

    for spk in unique_speakers:
        idx = (speakers == spk)
        spk_coords = coords[idx]

        # ---- scatter for legend ----
        plt.scatter(
            spk_coords[:, 0],
            spk_coords[:, 1],
            s=18,
            label=f"Spk {spk}"
        )

        # ---- annotation at cluster centroid ----
        cx = np.mean(spk_coords[:, 0])
        cy = np.mean(spk_coords[:, 1])

        ratio = contrast_stats[spk]['ratio_diff_gender']
        same = contrast_stats[spk]['same_gender']
        diff = contrast_stats[spk]['diff_gender']

        text = f"Spk {spk}\nDiff: {diff}\nSame: {same}\nRatio: {ratio:.2f}"

        plt.text(
            cx,
            cy,
            text,
            fontsize=9,
            weight="bold",
            bbox=dict(
                facecolor="white",
                alpha=0.7,
                edgecolor="black",
                boxstyle="round,pad=0.3"
            )
        )

    # ---- Add legend showing color mapping ----
    plt.legend(
        title="Speakers",
        fontsize=10,
        title_fontsize=11,
        loc="best"
    )

    plt.title(
        "t-SNE of Dual Speaker Embeddings\n(Annotated with Gender-Contrast Stats)",
        fontsize=14
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[✔] Saved annotated t-SNE plot to {save_path}")


# =====================================================================
# 8) MAIN
# =====================================================================

if __name__ == "__main__":

    #META = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/Libri2Mix/wav16k/min/metadata/mixture_dev_mix_clean.csv" #dev
    META= "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/Libriuni_05_08/Libri2Mix_ovl50to80/wav16k/min/metadata/mixture_train-360_mix_both.csv"
    CKPT = "/mnt/disks/data/model_ckpts/librispeech_asp_wavlm_ft_dualemb_queryorthogonality_mhqa_lib2mix_tr360_trfs_valwham/best-epoch=50-val_separation=0.000.ckpt"
    SPEAKER_TXT = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/LibriSpeech/SPEAKERS.TXT"

    SAVE_TSNE = "/home/sidharth./codebase/wavlm_dual_embedding/analysis/tsne/tsne_dual_train_mixboth_seed44_queryorthogonal_mhqa_ft_with_overlap_05_08_trained360.png"

    # ------ Load everything ------
    metadata = parse_metadata(META)
    gender_map = parse_speaker_gender(SPEAKER_TXT)
    model = load_dual_model(CKPT, device='cpu')

    # ------ Choose speakers ------
    chosen_speakers, valid_map = select_speakers(
        metadata, min_mixtures=20, num_speakers=4
    )
    print("Selected speakers:", chosen_speakers)

    speaker_files = {s: valid_map[s] for s in chosen_speakers}

    # ------ Extract embeddings + mixpaths ------
    embs, labels, mixpaths = extract_dual_embeddings(
        model, speaker_files, gender_map, device='cpu'
    )

    # ------ Compute gender contrast ------
    contrast_stats = compute_gender_contrast_per_speaker(
        labels, mixpaths, metadata, gender_map
    )

    # print("\n[✔] Gender contrast stats (per speaker):")
    # breakpoint()
    # print(json.dumps(contrast_stats, indent=4))

    # ------ Plot t-SNE ------
    plot_tsne(embs, labels, contrast_stats, SAVE_TSNE)

