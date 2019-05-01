from torch import nn
import torch.nn.functional as f
import numpy as np
import torch

class Model(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        
        # Network defition
        # self.pad = nn.ConstantPad1d(self.hparams.sample_rate // 160, 0)
        self.conv1 = nn.Conv1d(in_channels=1,
                               out_channels=15,
                               padding=self.hparams.sample_rate // 160,
                               kernel_size=self.hparams.sample_rate // 80)
        self.conv2 = nn.Conv1d(in_channels=15,
                               out_channels=15,
                               padding=70,
                               kernel_size=141)
        self.conv3 = nn.Conv1d(in_channels=15,
                               out_channels=15,
                               padding=18,
                               kernel_size=37)
        self.conv4 = nn.Conv1d(in_channels=15,
                               out_channels=1,
                               padding=5,
                               kernel_size=11)                      
    
    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)
        # Shape: (batch, channels, seq_length)

        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        
        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        
        x = self.conv4(x)
        
        x = x.squeeze()
        return x

    def spectrogram(self, wav):
        stft = torch.stft(
            wav, 
            n_fft=self.hparams.n_fft,
            hop_length=self.hparams.hop_size,
            win_length=self.hparams.win_size,
            window=torch.hann_window(self.hparams.win_size).cuda()
        )
        power = (stft ** 2).sum(dim=-1)
        log_spec = 10. * torch.log10(torch.clamp(power / power.max(), 1e-10))
        return torch.max(log_spec, log_spec.max() - self.hparams.top_db)

    def loss(self, y_pred, y_true):
        diff = self.spectrogram(y_pred) - self.spectrogram(y_true)
        # diff = y_pred - y_true
        l2_loss = torch.mean(diff ** 2)
        return l2_loss
