import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel, WavLMConfig
from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling


class MultiHeadQueryPooling(nn.Module):
    def __init__(self, input_dim = 768, output_dim = 256, num_queries=2, num_heads=4):

        super().__init__()
        self.input_dim = input_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads 
        assert input_dim % num_heads == 0, "output_dim must be divisible by num_heads"

        self.key_proj = nn.Linear(input_dim, input_dim)
        self.value_proj = nn.Linear(input_dim, input_dim)

        self.queries = nn.Parameter(torch.randn(1, num_heads, num_queries, self.head_dim))

        self.out_proj = nn.Linear(input_dim, output_dim)

        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.queries)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, return_queries=False):
        """
        x: [B, T, D=768]
        Returns:
            pooled: [B, num_queries=nsp, output_dim]
        """
     
        B, T, D = x.shape

        # Project keys and values
        K = self.key_proj(x).view(B,T, self.num_heads, self.head_dim)    # [B, T, nH, HD]
        V = self.value_proj(x).view(B,T, self.num_heads, self.head_dim)  # [B, T, nH, HD]

        #Transpose for attention computation
        K = K.permute(0,2,1,3)  # [B, nH, T, HD]
        V = V.permute(0,2,1,3)  # [B, nH, T, HD]

        #prepare queries
        Q = self.queries.expand(B, -1, -1, -1)  # [B, nH, nQ, HD]

        #calculte attention scores
        scores = torch.einsum('bhqd, bhdt -> bhqt', Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)  # [B, nH, nQ, T]
        # attn_weights = F.softmax(scores, dim=-1)  # [B, nH, nQ, T]
        #normalize across query dimension
        attn_weights = F.softmax(scores, dim=-1) #[B, nH, nQ, T]
        
        #optionally time as well
        attn_q = attn_weights / (attn_weights.sum(dim=-2, keepdim=True) + 1e-6)

        #Aggregate values

        emb_heads = torch.einsum('bhqt, bhtd -> bhqd', attn_q, V)  # [B, nH, nQ, HD]

        emb = emb_heads.permute(0,2,1,3).reshape(B, self.num_queries, D)  # [B, nQ, D]
        emb = self.out_proj(emb)  # [B, nQ, output_dim]
        
        if return_queries:
            return emb, self.queries, attn_q

        return emb



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


class SpeakerEncoderDualWrapper(nn.Module):
    """
    For Phase 1: this is actually a speaker encoder
    using WavLM + projection + ASP.
    """
    def __init__(self, emb_dim=256, finetune_wavlm=False):
        super().__init__()

        # Load WavLM
        config = WavLMConfig.from_pretrained("microsoft/wavlm-base-plus")
        self.wavlm = WavLMModel.from_pretrained(
            "microsoft/wavlm-base-plus",
            config=config,
            ignore_mismatched_sizes=True
        )
        self.wavlm.requires_grad_(False)  # freeze full wavlm

        if finetune_wavlm == True:
            for layer in self.wavlm.encoder.layers[-6:]:
                for param in layer.parameters():
                    param.requires_grad = True
            
            for param in self.wavlm.encoder.layer_norm.parameters():
                param.requires_grad = True
            
            for param in self.wavlm.encoder.pos_conv_embed.parameters():
                param.requires_grad = True

            print("Unfreezing last 6 layers of WavLM")

        self.wavlm_out = 768
        self.emb_dim = emb_dim

        # Multi-head query pooling to get 2 streams
        self.mhqp = MultiHeadQueryPooling(input_dim=self.wavlm_out, output_dim = self.emb_dim, num_queries=2, num_heads=4)

    def forward(self, audio, return_queries=False):
        """
        mix_audio: [B, T]
        """
        if audio.dim() == 3:  # [B, 1, T]
            audio = audio.squeeze(1)

        # WavLM gives [B, T_frames, 768]
        feats = self.wavlm(audio).last_hidden_state   # [B, T, 768]

        #multi-head query pooling to get 2 streams
        if return_queries:
            emb, Q, attn_scores  = self.mhqp(feats, return_queries=True)
            return emb, Q, attn_scores

        emb = self.mhqp(feats)  # [B, 2, emb_dim]
        

        return emb


if __name__ == "__main__":
    breakpoint()
    model = SpeakerEncoderDualWrapper(emb_dim=256)
    dummy_audio = torch.randn(2, 16000 * 3)  #
    emb = model(dummy_audio)
    print(emb.shape)  # should be [2, 256]
    