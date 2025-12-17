from resemblyzer.hparams import *
from torch import nn
import torch
import torchaudio
from pathlib import Path
from typing import Union


class VoiceEncoder(nn.Module):
    def __init__(self, weights_fpath: Union[Path, str] = None):
        """
        Speaker encoder that can take either:
          - raw waveforms:  (B, T)
          - mel spectrogram: (B, T, mel_n_channels)

        and returns:
          - embeddings: (B, model_embedding_size), L2-normalized
        """
        super().__init__()

        # --------- Torch mel frontend ----------
        n_fft = int(sampling_rate * mel_window_length / 1000)
        hop_length = int(sampling_rate * mel_window_step / 1000)

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=mel_n_channels,
            power=2.0,       # like librosa's power mel
            center=True,
        )

        # --------- Encoder backbone ----------
        self.lstm = nn.LSTM(
            input_size=mel_n_channels,
            hidden_size=model_hidden_size,
            num_layers=model_num_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()

        # Optional: load pretrained weights
        if weights_fpath is not None:
            weights_fpath = Path(weights_fpath)
            checkpoint = torch.load(weights_fpath, map_location="cpu")
            # assumes checkpoint["model_state"] matches lstm+linear keys
            self.load_state_dict(checkpoint["model_state"], strict=False)

    def _wav_to_mel_torch(self, wav_batch: torch.Tensor) -> torch.Tensor:
        """
        :param wav_batch: (B, T) or (T,) raw waveform in float32/float64
        :return: mels: (B, T_mel, mel_n_channels)
        """
        # Ensure batch dimension
        if wav_batch.dim() == 1:
            wav_batch = wav_batch.unsqueeze(0)  # (1, T)

        # Torchaudio expects (B, T)
        # -> output: (B, n_mels, T_mel)
        mels = self.mel_spec(wav_batch)        # (B, mel_n_channels, T_mel)
        mels = mels.transpose(1, 2)           # (B, T_mel, mel_n_channels)
        return mels

    def _encode_mel(self, mels: torch.Tensor) -> torch.Tensor:
        """
        :param mels: (B, T, mel_n_channels)
        :return: embeds: (B, model_embedding_size)
        """
        _, (hidden, _) = self.lstm(mels)             # hidden: (num_layers, B, H)
        embeds_raw = self.relu(self.linear(hidden[-1]))  # (B, D)
        embeds = embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)
        return embeds

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unified forward:

        - If x is (B, T): treat as raw waveforms → compute mel inside.
        - If x is (B, T, C): assume it's already mel → use directly.

        Always returns:
          - (B, model_embedding_size)
        """
        if x.dim() == 2:
            # x: (B, T) raw waves
            mels = self._wav_to_mel_torch(x)  # (B, T_mel, C)
        elif x.dim() == 3:
            # x: (B, T, C) mels already
            mels = x
        else:
            raise ValueError(f"Expected input shape [B, T] or [B, T, C], got {x.shape}")

        return self._encode_mel(mels)