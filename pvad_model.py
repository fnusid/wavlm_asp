import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torchaudio
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
    

class pVAD_module(nn.Module):
    def __init__(self, hidden_dim=256+80):
        super(pVAD_module, self).__init__()
        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80), #25ms window, 10ms hop, 80 mel bins
            )
        

        self.lstm_layers = nn.LSTM(input_size=hidden_dim, hidden_size=64, num_layers=2, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(64, 1)

    def forward(self, x, emb=None):
        # x: [B, T]
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True)
        #x is now [B, feat, T]
        #emb is [B, 256]
        if emb is None:
            raise ValueError("Speaker embedding is required for pVAD_module.")
        x = torch.cat([x, emb.unsqueeze(-1).expand(-1,-1, x.size(-1))], dim=1)  # [B, feat+256, T]
        x = x.transpose(1, 2)  # [B, T, feat+256]
        lstm_out, _ = self.lstm_layers(x)  # lstm_out: [B, T, 64]
        logits = self.fc(lstm_out)         # logits: [B, T, 1]
        return logits.squeeze(-1)          # [B, T]
    




if __name__ == "__main__":
    # Test the pVAD_module
    batch_size = 4
    time_steps = 100
    feature_dim = 256   
    breakpoint()

    dummy_input = torch.randn(batch_size, time_steps, feature_dim)
    model = pVAD_module(hidden_dim=feature_dim)
    output = model(dummy_input)

    print("Output shape:", output.shape)  # Should be [B, T]