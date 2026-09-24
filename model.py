import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel, WavLMConfig
# from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling
import torchaudio
import math


class CausalConv1D(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, dilation):
        super().__init__()
        self.padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation)
    
    def forward(self, x):
        x_pad = F.pad(x, (self.padding, 0), mode="constant", value=0)
        x = self.conv(x_pad)
        return x

        

class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, input):
        x = self.se(input)
        return input * x


def Cumsum_Pool(x):
    T = x.size(-1)
    count = torch.arange(1, T + 1, device=x.device)

    y = x.cumsum(dim=-1) / count  # [B, C, T]
    return y


class Causal_SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(Causal_SEModule, self).__init__()
        self.se = nn.Sequential(
            CausalConv1D(channels, bottleneck, 1, 1, 1),
            nn.ReLU(),
            CausalConv1D(bottleneck, channels, 1, 1, 1),
            nn.Sigmoid(),
        )
    def forward(self, input):
        x = Cumsum_Pool(input) #[B, C, T]
        x = self.se(x)
        return input * x

class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale = 8):
        super(Bottle2neck, self).__init__()
        width       = int(math.floor(planes / scale))
        # self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.conv1 = CausalConv1D(inplanes, width*scale, kernel_size=1, stride=1, dilation=1)
        # self.bn1    = nn.BatchNorm1d(width*scale)
        self.ln1 = nn.LayerNorm(width*scale)
        self.nums   = scale -1
        convs       = []
        # bns         = []
        lns         = []
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            # convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            convs.append(CausalConv1D(width, width, kernel_size, 1, dilation))
            # bns.append(nn.BatchNorm1d(width))
            lns.append(nn.LayerNorm(width))
        self.convs  = nn.ModuleList(convs)
        # self.bns    = nn.ModuleList(bns)
        self.lns = nn.ModuleList(lns)
        # self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.conv3 = CausalConv1D(width*scale, planes, 1,1, 1)
        # self.bn3    = nn.BatchNorm1d(planes)
        self.ln3 = nn.LayerNorm(planes)
        self.relu   = nn.ReLU()
        self.width  = width
        # self.se     = SEModule(planes)
        self.se = Causal_SEModule(planes)

    def forward(self, x):
        #x : [B, C, T]
     
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        #reshape it 
        out = out.transpose(1, 2) #[B, T, C]
        out = self.ln1(out) 
        #reshape it back
        out = out.transpose(2, 1) #[B, C, T]

        spx = torch.split(out, self.width, 1) # 8 arrays of [B, C/8, T]
        for i in range(self.nums): #7
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(sp)
          #reshape
          sp = sp.transpose(1, 2)
          sp = self.lns[i](sp)
          sp = sp.transpose(1,2 )
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        out = self.relu(out)
        out = out.transpose(1, 2)
        out = self.ln3(out)
        out = out.transpose(1, 2)
        
        out = self.se(out)
        out += residual
        return out 

class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        # breakpoint()
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)

class FbankAug(nn.Module):

    def __init__(self, freq_mask_width = (0, 8), time_mask_width = (0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)
            
        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):    
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x

class ECAPA_TDNN_encoder(nn.Module):

    def __init__(self, C):

        super(ECAPA_TDNN_encoder, self).__init__()

        n_fft, hop_length = 512, 160
        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=n_fft, win_length=400, hop_length=hop_length, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80,
                                                 center=False),
            )
        # center=False makes each STFT frame look only at past+current audio;
        # left-pad by n_fft - hop_length so it still gets a frame near t=0
        self.stft_pad = n_fft - hop_length

        self.specaug = FbankAug() # Spec augmentation
        self.d_model = 1536
        # self.conv1  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.conv1 = CausalConv1D(80, C, kernel_size=5, stride=1, dilation=1)
        self.relu   = nn.ReLU()
        # self.bn1    = nn.BatchNorm1d(C)
        self.ln1 = nn.LayerNorm(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3*C, self.d_model, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            # nn.BatchNorm1d(256),
            nn.LayerNorm(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, self.d_model, kernel_size=1),
            nn.Softmax(dim=2),
            )
        # self.bn5 = nn.BatchNorm1d(3072)
        self.ln5 = nn.LayerNorm(3072)
        self.fc6 = nn.Linear(3072, 256)
        # self.bn6 = nn.BatchNorm1d(256)
        self.ln6 = nn.LayerNorm(256)

    def forward(self, x, aug=False):
        with torch.no_grad():
            x = F.pad(x, (self.stft_pad, 0), mode="reflect")
            x = self.torchfbank(x)+1e-6
            x = x.log()
            x = x - Cumsum_Pool(x)  # causal running mean, instead of the full-utterance mean
            if aug == True:
                x = self.specaug(x)
 
        # x shape [B, 80, T], 80 : Mel spec features
        x = self.conv1(x) #[B, C, T]
       
        x = self.relu(x) #[B, C, T]
        #reshape it to to [B, T, C] 
        x = torch.transpose(x, 2, 1)
        x = self.ln1(x) #[B, T, C]
        x = torch.transpose(x, 2, 1)
        #reshape it back to [B, C, T]

        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x) #[B, 1536, T]

        return x




class Causal_AttentiveStatisticsPooling(nn.Module):
    """
    Streaming ASP: mean/std and attention weights at time t depend only on
    frames 0..t. Returns a running [mean; std] per frame instead of a single
    pooled vector for the whole utterance.
    """
    def __init__(self, channels, attention_channels=128, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.tdnn = CausalConv1D(channels * 3, attention_channels, kernel_size=1, stride=1, dilation=1)
        self.tanh = nn.Tanh()
        self.conv = CausalConv1D(attention_channels, channels, kernel_size=1, stride=1, dilation=1)

    def forward(self, x):
        # x: [B, C, T]
        run_mean = Cumsum_Pool(x)  # causal running mean, [B, C, T]
        run_var = (Cumsum_Pool(x.pow(2)) - run_mean.pow(2)).clamp(min=self.eps)
        run_std = torch.sqrt(run_var)

        attn_in = torch.cat([x, run_mean, run_std], dim=1)  # [B, 3C, T]
        logits = self.conv(self.tanh(self.tdnn(attn_in)))   # [B, C, T]
        # clamp instead of subtracting a running max: a per-t max would break
        # the cumsum decomposition used below
        logits = logits.clamp(-15, 15)
        w = torch.exp(logits)

        num1 = torch.cumsum(w * x, dim=-1)
        num2 = torch.cumsum(w * x.pow(2), dim=-1)
        den = torch.cumsum(w, dim=-1)

        mean = num1 / den
        var = (num2 / den - mean.pow(2)).clamp(min=self.eps)
        std = torch.sqrt(var)

        return torch.cat([mean, std], dim=1)  # [B, 2C, T]


class SpeakerEncoder(nn.Module):
    def __init__(self, feat_dim, emb_dim=256, streaming=False):
        super().__init__()
        # self.asp = AttentiveStatisticsPooling(feat_dim)
        self.asp = Causal_AttentiveStatisticsPooling(feat_dim)
        self.linear = nn.Linear(feat_dim * 2, emb_dim)
        self.streaming = streaming

    def forward(self, x):
        """
        x: [B, D, T] (projected features)
        streaming=True  -> [B, T, emb_dim], a running embedding per frame
        streaming=False -> [B, emb_dim], one embedding per utterance (final
                            frame of the running stats == the full-utterance
                            causal stats, since it has seen every frame)
        """
        pooled = self.asp(x)  # [B, 2D, T]
        if self.streaming:
            pooled = pooled.transpose(1, 2)   # [B, T, 2D]
        else:
            pooled = pooled[:, :, -1]         # [B, 2D]
        emb = self.linear(pooled)
        return F.normalize(emb, p=2, dim=-1)


class SpeakerEncoderDualWrapper(nn.Module):
    """
    For Phase 1: this is actually a speaker encoder
    using WavLM + projection + ASP.
    """
    def __init__(self, emb_dim=256, streaming=False):
        super().__init__()

        # Load WavLM

        # self.encoder = ECAPA_TDNN_encoder(C=2048)
        # self.encoder = ECAPA_TDNN_encoder(C=1024)
        self.encoder = ECAPA_TDNN_encoder(C=3072)
        self.emb_dim = emb_dim
        self.streaming = streaming

        # Linear 768 -> 256
        self.projector = nn.Linear(self.encoder.d_model, 2*emb_dim)

        # ASP-based speaker encoder
        self.encoder1 = SpeakerEncoder(feat_dim=emb_dim, emb_dim=emb_dim, streaming=streaming)
        self.encoder2 = SpeakerEncoder(feat_dim=emb_dim, emb_dim=emb_dim, streaming=streaming)

    def forward(self, audio):
        """
        mix_audio: [B, T]
        """
        if audio.dim() == 3:  # [B, 1, T]
            audio = audio.squeeze(1)

        # WavLM gives [B, T_frames, 768]
        # feats = self.wavlm(audio).last_hidden_state   # [B, T, 768]
        feats = self.encoder(audio).transpose(1, 2)   # [B, T, 1536]

        # Project to smaller dimension
        proj = self.projector(feats)   # [B, T, 512]

        # Split for dual embedding
        proj1, proj2 = torch.chunk(proj, 2, dim=-1)  # each [B, T, 256]


        # ASP expects [B, D, T]
        proj1 = proj1.transpose(1, 2)    # [B, 256, T]
        proj2 = proj2.transpose(1, 2)    # [B, 256, T]

        # Get speaker embedding: [B, T, 256] per-frame if streaming,
        # else [B, 256] one embedding per utterance
        emb1 = self.encoder1(proj1)
        emb2 = self.encoder2(proj2)
        stack_dim = 2 if self.streaming else 1
        emb = torch.stack([emb1, emb2], dim=stack_dim)  # [B,T,2,256] or [B,2,256]
        return emb


if __name__ == "__main__":

    model = SpeakerEncoderDualWrapper(emb_dim=256)
    dummy_audio = torch.randn(2, 16000 * 3)  #
    emb = model(dummy_audio)
    print(emb.shape)  # should be [2, n_emb, 256]
    