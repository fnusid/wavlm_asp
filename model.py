import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel, WavLMConfig
from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling
import torchaudio
import math

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

class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale = 8):
        super(Bottle2neck, self).__init__()
        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(width*scale)
        self.nums   = scale -1
        convs       = []
        bns         = []
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(planes)
        self.relu   = nn.ReLU()
        self.width  = width
        self.se     = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(sp)
          sp = self.bns[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        
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

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80), #25ms window, 10ms hop, 80 mel bins
            )

        self.specaug = FbankAug() # Spec augmentation
        self.d_model = 1536
        self.conv1  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3*C, self.d_model, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, self.d_model, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 256)
        self.bn6 = nn.BatchNorm1d(256)

    def extract_logmel(self, x, aug=False):
        '''
        x: waveform [B, T]
        returns: log-mel spectrogram [B, 80, T_frames]

        '''
        with torch.no_grad():
            x = self.torchfbank(x) + 1e-6
            x = x.log()
            x = x - torch.mean(x, dim=-1, keepdim=True)

            if aug:
                x = self.specaug(x)

        return x

    def forward_features(self, x):
        '''
        x: log-mel spectrogram [B, 80, T_frames]
        '''
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x) #[B, 1536, T]
        return x

    # def forward(self, x, aug=False):
    #     with torch.no_grad():
    #         x = self.torchfbank(x)+1e-6
    #         x = x.log()   
    #         x = x - torch.mean(x, dim=-1, keepdim=True)
    #         if aug == True:
    #             x = self.specaug(x)
    #     # breakpoint() #x: [B, feat, T]
    #     x = self.conv1(x)
    #     x = self.relu(x)
    #     x = self.bn1(x)

    #     x1 = self.layer1(x)
    #     x2 = self.layer2(x+x1)
    #     x3 = self.layer3(x+x1+x2)

    #     x = self.layer4(torch.cat((x1,x2,x3),dim=1))
    #     x = self.relu(x) #[B, 1536, T]

    #     return x

    def forward(self, x, aug=False):
        x = self.extract_logmel(x, aug=aug)
        x = self.forward_features(x)
        return x



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
    def __init__(self, emb_dim=256, vad_hidden=128):
        super().__init__()

        # Load WavLM
        self.encoder = ECAPA_TDNN_encoder(C=3072)
        # self.encoder = ECAPA_TDNN_encoder(C=2048)
        # self.encoder = ECAPA_TDNN_encoder(C=1024)
       
        self.emb_dim = emb_dim

        # Linear 768 -> 256
        self.projector = nn.Linear(self.encoder.d_model, 2*emb_dim)

        # ASP-based speaker encoder
        self.encoder1 = SpeakerEncoder(feat_dim=emb_dim, emb_dim=emb_dim)
        self.encoder2 = SpeakerEncoder(feat_dim=emb_dim, emb_dim=emb_dim)

        # self.vad_head1 = nn.Sequential(
        #     nn.Conv1d(emb_dim, vad_hidden, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv1d(vad_hidden, 1, kernel_size=1),
        # )
        # self.vad_head2 = nn.Sequential(
        #     nn.Conv1d(emb_dim, vad_hidden, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv1d(vad_hidden, 1, kernel_size=1),
        # )
    def forward_from_features(self, feats):
        """
        feats: [B, T_frames, feat_dim]
        returns: [B, 2, emb_dim]
        """
        feats = self.encoder.forward_features(feats).transpose(1, 2)   # [B, T, 1536]

        proj = self.projector(feats)   # [B, T, 512]

        proj1, proj2 = torch.chunk(proj, 2, dim=-1)  # each [B, T, 256]
        proj1 = proj1.transpose(1, 2)    # [B, 256, T]
        proj2 = proj2.transpose(1, 2)    # [B,

        emb1 = self.encoder1(proj1)       # [B, 256]
        emb2 = self.encoder2(proj2)       # [B, 256]
        emb = torch.stack([emb1, emb2], dim=1) #[B,2,256]
        return emb
    
    def forward(self, audio):
        '''
        audio: [B, T]
        '''
        if audio.ndim== 3:  # [B, 1, T]
            audio = audio.squeeze(1)
        feats = self.encoder.extract_logmel(audio, aug=False)
        emb = self.forward_from_features(feats)
        return emb

    # def forward(self, audio, return_vad=False):
    #     """
    #     mix_audio: [B, T]
    #     """
    #     if audio.dim() == 3:  # [B, 1, T]
    #         audio = audio.squeeze(1)

    #     # WavLM gives [B, T_frames, 768]
    #     # feats = self.wavlm(audio).last_hidden_state   # [B, T, 768]
    #     feats = self.encoder(audio).transpose(1, 2)   # [B, T, 1536]

    #     # Project to smaller dimension
    #     proj = self.projector(feats)   # [B, T, 512]

    #     # Split for dual embedding
    #     proj1, proj2 = torch.chunk(proj, 2, dim=-1)  # each [B, T, 256]


    #     # ASP expects [B, D, T]
    #     proj1 = proj1.transpose(1, 2)    # [B, 256, T]
    #     proj2 = proj2.transpose(1, 2)    # [B, 256, T]

    #     # Get speaker embedding
    #     emb1 = self.encoder1(proj1)       # [B, 256]
    #     emb2 = self.encoder2(proj2)       # [B, 256]
    #     emb = torch.stack([emb1, emb2], dim=1) #[B,2,256]

    #     # if not return_vad:
        
    #     #     return emb

    #     # else:
    #     #     vad1 = self.vad_head1(proj1)
    #     #     vad2 = self.vad_head2(proj2)
    #     #     vad_logits = torch.stack([vad1, vad2], dim=1)   # [B, 2, T]
    #     #     return emb, vad_logits
    #     return emb


if __name__ == "__main__":

    model = SpeakerEncoderDualWrapper(emb_dim=256)
    dummy_audio = torch.randn(2, 16000 * 3)  #
    emb = model(dummy_audio)
    print(emb.shape)  # should be [2, n_emb, 256]
    