import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
import webrtcvad

# -----------------------------
# Constants
# -----------------------------
RATE = 16000
HOP_SAMPLES = 160        # matches MelSpectrogram hop_length=160 (10 ms @ 16k)
VAD_FRAME_MS = 10        # 10ms so VAD frames align with mel-frame hop
EPS = 1e-8

# WebRTC VAD (0..3 aggressiveness)
VAD = webrtcvad.Vad(2)


# -----------------------------
# Noise mixing
# -----------------------------
def mix_noise_with_snr(clean, noise, snr_db):
    """
    clean, noise: torch tensors [T] or [1,T]
    snr_db: desired SNR (clean_power / noise_power in dB)
    """
    if noise.ndim > 1:
        noise = noise.mean(0)
    if clean.ndim > 1:
        clean = clean.mean(0)

    if len(noise) < len(clean):
        diff = len(clean) - len(noise)
        noise = F.pad(noise, (diff // 2, diff - (diff // 2)))
    else:
        noise = noise[: len(clean)]

    clean_power = clean.pow(2).mean()
    noise_power = noise.pow(2).mean()

    target_noise_power = clean_power / (10 ** (snr_db / 10))
    scale = torch.sqrt(target_noise_power / (noise_power + EPS))

    return clean + scale * noise


# -----------------------------
# VAD helpers (frame-level labels at 10ms)
# -----------------------------
def vad_labels_10ms(audio_f32_1d: np.ndarray, sr=RATE, frame_ms=VAD_FRAME_MS) -> np.ndarray:
    """
    audio_f32_1d: float32 in [-1,1], shape [T]
    returns: np.uint8 labels shape [n_frames], 1=speech, 0=non-speech
    """
    assert sr in (8000, 16000, 32000, 48000)
    assert frame_ms in (10, 20, 30)

    if audio_f32_1d.ndim != 1:
        audio_f32_1d = audio_f32_1d.reshape(-1)

    frame_len = int(sr * frame_ms / 1000)  # 160 samples for 10ms @16k
    n_frames = len(audio_f32_1d) // frame_len
    if n_frames <= 0:
        return np.zeros((0,), dtype=np.uint8)

    pcm16 = np.clip(audio_f32_1d, -1.0, 1.0)
    pcm16 = (pcm16 * 32768.0).astype(np.int16).tobytes()

    labels = np.zeros(n_frames, dtype=np.uint8)
    for i in range(n_frames):
        start = i * frame_len * 2
        frame = pcm16[start: start + frame_len * 2]
        labels[i] = 1 if VAD.is_speech(frame, sr) else 0

    return labels


def fit_vad_to_T(vad: np.ndarray, T: int) -> np.ndarray:
    """
    Crop/pad a VAD label sequence to exactly length T.
    """
    vad = np.asarray(vad, dtype=np.uint8)
    if vad.shape[0] >= T:
        return vad[:T]
    pad = np.zeros((T - vad.shape[0],), dtype=np.uint8)
    return np.concatenate([vad, pad], axis=0)


# -----------------------------
# Dataset
# -----------------------------
class MyLibri2Mix(Dataset):
    """
    Returns:
      mix_audio:  [T] float32
      sources:    [2, T] float32
      spk_labels: [2] long
      vad_src:    [2, T_frames] float32  (10ms VAD per source, aligned to hop=160)
    """
    def __init__(
        self,
        metadata_path,
        speaker_map_path,
        num_speakers=2,
        sampling_rate=RATE,
        split="train",
        noise_prob=0.5,
    ):
        super().__init__()
        self.metadata = pd.read_csv(metadata_path)
        self.num_speakers = num_speakers
        self.sampling_rate = sampling_rate
        self.noise_prob = noise_prob

        if split == "train":
            self.noise_file_path = "/home/sidcs/datasets/LibriMix/LibriMix/noise_files_embedding_model/freesound_noise_bins.json"
        else:
            self.noise_file_path = "/home/sidcs/datasets/LibriMix/LibriMix/noise_files_embedding_model/wham_tt_noise_bins.json"

        with open(self.noise_file_path, "r") as f:
            self.noise_dict = json.load(f)

        with open(speaker_map_path, "r") as f:
            self.speaker_to_index = json.load(f)

    def __len__(self):
        return len(self.metadata)

    def _pick_noise_file_for_length(self, mix_len_sec: int):
        bins = list(self.noise_dict.keys())
        if len(bins) == 0:
            return None

        uppers = np.array([int(b.split("-")[1]) for b in bins], dtype=np.int32)
        cand = uppers[uppers > mix_len_sec]
        ub = int(uppers.max() if len(cand) == 0 else cand[0])

        candidate_keys = []
        if ub in [5, 10]:
            candidate_keys.append(f"{ub-5}-{ub}")
        else:
            candidate_keys.append(f"{ub-10}-{ub}")

        valid_key = None
        for k in candidate_keys:
            if k in self.noise_dict and len(self.noise_dict[k]) > 0:
                valid_key = k
                break

        if valid_key is None:
            for k in bins:
                if len(self.noise_dict[k]) > 0:
                    valid_key = k
                    break

        if valid_key is None:
            return None

        return random.choice(self.noise_dict[valid_key])

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # ---- Mixture ----
        mix_path = row["mixture_path"]
        mix_audio, _ = torchaudio.load(mix_path)  # [C,T]
        mix_audio = mix_audio.mean(0)             # [T]

        # ---- Sources ----
        source_audios = []
        for i in range(self.num_speakers):
            s_path = row[f"source_{i+1}_path"]
            s_audio, _ = torchaudio.load(s_path)
            s_audio = s_audio.mean(0)             # [T]
            source_audios.append(s_audio)

        # (Optional) Noise on mixture only
        if random.random() < self.noise_prob:
            mix_len_sec = int(mix_audio.shape[-1] // self.sampling_rate)
            noise_file = self._pick_noise_file_for_length(mix_len_sec)
            if noise_file is not None:
                noise_audio, _ = torchaudio.load(noise_file)
                r = random.random()
                if r < 0.4:
                    snr = random.uniform(-5, 5)
                elif r < 0.8:
                    snr = random.uniform(5, 15)
                else:
                    snr = random.uniform(15, 25)
                mix_audio = mix_noise_with_snr(mix_audio, noise_audio, snr)

        # Stack sources to [2,T]
        # (Assumes your dataset generation already ensured same length across mix/src)
        sources_tensor = torch.stack(source_audios, dim=0)  # [2, T]

        # Speaker labels [2]
        speaker_indices = []
        for i in range(self.num_speakers):
            speaker_id = str(row[f"speaker_{i+1}_ID"])
            if speaker_id in self.speaker_to_index:
                index = self.speaker_to_index[speaker_id]
            else:
                index = int(speaker_id)
            speaker_indices.append(index)
        labels_tensor = torch.tensor(speaker_indices, dtype=torch.long)

        # ---- Per-source VAD @ 10ms -> align to model frame axis ----
        # Define the frame axis using mixture length (consistent with hop=160)
        T_samples = int(mix_audio.numel())
        T_frames = T_samples // HOP_SAMPLES  # 10ms frames

        vad_src = []
        for i in range(self.num_speakers):
            s_np = source_audios[i].detach().cpu().numpy().astype(np.float32)
            vad_i = vad_labels_10ms(s_np, sr=self.sampling_rate, frame_ms=VAD_FRAME_MS)  # [Tv_i]
            vad_i = fit_vad_to_T(vad_i, T_frames)  # [T_frames]
            vad_src.append(vad_i)

        vad_src = np.stack(vad_src, axis=0)                 # [2, T_frames]
        vad_tensor = torch.from_numpy(vad_src).float()      # [2, T_frames]

        return mix_audio, sources_tensor, labels_tensor, vad_tensor


# -----------------------------
# Collate: pad mix/sources AND pad vad to max frames
# -----------------------------
def librimix_collate(batch):
    mix, source, labels, vad = zip(*batch)
    # mix:    list of [T]
    # source: list of [2,T]
    # vad:    list of [2,T_frames]

    mix_padded = pad_sequence(mix, batch_first=True, padding_value=0.0)  # [B, Tmax]

    # sources: pad on time. pad_sequence expects [T, C], so transpose [2,T] -> [T,2]
    sources_T2 = [s.transpose(0, 1) for s in source]  # [T,2]
    sources_padded = pad_sequence(sources_T2, batch_first=True, padding_value=0.0)  # [B, Tmax, 2]
    sources_padded = sources_padded.transpose(1, 2)  # [B, 2, Tmax]

    labels = torch.stack(labels, dim=0)  # [B,2]

    # vad: pad on time. pad_sequence expects [T_frames, 2], so transpose [2,Tf] -> [Tf,2]
    vad_T2 = [v.transpose(0, 1) for v in vad]  # [T_frames, 2]
    vad_padded = pad_sequence(vad_T2, batch_first=True, padding_value=0.0)  # [B, Tframes_max, 2]
    vad_padded = vad_padded.transpose(1, 2)  # [B, 2, Tframes_max]

    return mix_padded, sources_padded, labels, vad_padded


# -----------------------------
# Lightning DataModule
# -----------------------------
class LibriMixDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root,
        speaker_map_path,
        batch_size=32,
        num_workers=0,
        num_speakers=2,
        sample_rate=RATE,
    ):
        super().__init__()
        self.data_root = data_root
        self.speaker_map_path = speaker_map_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_speakers = num_speakers
        self.sample_rate = sample_rate
        self.persistent_workers = True if self.num_workers > 0 else False

        self.base_data_path = os.path.join(
            self.data_root,
            "Libriuni_05_08/Libri2Mix_ovl50to80/wav16k/min",
        )
        self.metadata_path = os.path.join(self.base_data_path, "metadata")

    def setup(self, stage=None):
        train_meta = os.path.join(self.metadata_path, "mixture_train-360_mix_clean.csv")
        val_meta = os.path.join(self.metadata_path, "mixture_dev_mix_clean.csv")
        test_meta = os.path.join(self.metadata_path, "mixture_test_mix_clean.csv")

        self.train_dataset = MyLibri2Mix(
            metadata_path=train_meta,
            speaker_map_path=self.speaker_map_path,
            num_speakers=self.num_speakers,
            sampling_rate=self.sample_rate,
            split="train",
        )
        self.val_dataset = MyLibri2Mix(
            metadata_path=val_meta,
            speaker_map_path=self.speaker_map_path,
            num_speakers=self.num_speakers,
            sampling_rate=self.sample_rate,
            split="val",
        )
        self.test_dataset = MyLibri2Mix(
            metadata_path=test_meta,
            speaker_map_path=self.speaker_map_path,
            num_speakers=self.num_speakers,
            sampling_rate=self.sample_rate,
            split="test",
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=librimix_collate,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=librimix_collate,
            persistent_workers=self.persistent_workers,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=librimix_collate,
            persistent_workers=self.persistent_workers,
        )


# -----------------------------
# Quick sanity test
# -----------------------------
if __name__ == "__main__":
    data_root = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix"
    speaker_map_path = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/Libriuni_03_08/Libri2Mix_ovl30to80/wav16k/min/metadata/train360_mapping.json"

    dm = LibriMixDataModule(
        data_root=data_root,
        speaker_map_path=speaker_map_path,
        batch_size=4,
        num_workers=0,
        num_speakers=2,
        sample_rate=16000,
    )
    dm.setup()

    dl = dm.train_dataloader()
    mix, sources, spk_labels, vad = next(iter(dl))

    print("mix:", mix.shape)           # [B, Tmax]
    print("sources:", sources.shape)   # [B, 2, Tmax]
    print("labels:", spk_labels.shape) # [B, 2]
    print("vad:", vad.shape)           # [B, 2, Tframes_max]
    print("Tmax/160 approx:", mix.shape[-1] // 160, "  vad_max:", vad.shape[-1])