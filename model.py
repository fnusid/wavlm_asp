import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel, WavLMConfig
from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling







class SpeakerEncoder(nn.Module):
    def __init__(self, feat_dim, emb_dim=256):
        super().__init__()
        self.asp = AttentiveStatisticsPooling(feat_dim)
        self.linear = nn.Linear(feat_dim * 2, emb_dim)

    def forward(self, x):
        """
        x: [B, D, T] (projected features)
        """
        pooled = self.asp(x).squeeze(-1)  # [B, 2D]
        emb = self.linear(pooled)         # [B, emb_dim]
        return F.normalize(emb, p=2, dim=-1)


class SpeakerEncoderWrapper(nn.Module):
    """
    For Phase 1: this is actually a speaker encoder
    using WavLM + projection + ASP.
    """
    def __init__(self, emb_dim=256):
        super().__init__()

        # Load WavLM
        config = WavLMConfig.from_pretrained("microsoft/wavlm-base-plus")
        self.wavlm = WavLMModel.from_pretrained(
            "microsoft/wavlm-base-plus",
            config=config,
            ignore_mismatched_sizes=True
        )
        self.wavlm.requires_grad_(False)  # freeze full wavlm

        self.wavlm_out = 768
        self.emb_dim = emb_dim

        # Linear 768 -> 256
        self.projector = nn.Linear(self.wavlm_out, emb_dim)

        # ASP-based speaker encoder
        self.encoder = SpeakerEncoder(feat_dim=emb_dim, emb_dim=emb_dim)

    def forward(self, audio):
        """
        audio: [B, T]
        """
        if audio.dim() == 3:  # [B, 1, T]
            audio = audio.squeeze(1)

        # WavLM gives [B, T_frames, 768]
        feats = self.wavlm(audio).last_hidden_state   # [B, T, 768]

        # Project to smaller dimension
        proj = self.projector(feats)   # [B, T, 256]

        # ASP expects [B, D, T]
        proj = proj.transpose(1, 2)    # [B, 256, T]

        # Get speaker embedding
        emb = self.encoder(proj)       # [B, 256]
        return emb


if __name__ == "__main__":
    model = SpeakerEncoderWrapper(emb_dim=256)
    dummy_audio = torch.randn(2, 16000 * 3)  #
    emb = model(dummy_audio)
    print(emb.shape)  # should be [2, 256]
    