from torch import nn
import numpy as np
import torch


class Model(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        
        # Network defition
        self.conv1 = nn.Conv1d(in_channels=1,
                               out_channels=1,
                               kernel_size=1,
                               bias=False)
        
        self.conv1.weight.data[...] = -0.5       # For testing the identity
        
        # self.lstm = nn.LSTM(input_size=mel_n_channels,
        #                     hidden_size=model_hidden_size,
        #                     num_layers=model_num_layers,
        #                     batch_first=True).to(device)
        # self.relu = torch.nn.ReLU().to(device)
        
    
    def forward(self, x: torch.Tensor):
        # Obtain a tensor of shape (batch, channels, seq_length)
        x = x.unsqueeze(1)
        
        # Single unit for predicting the identity
        x = self.conv1(x)
        
        # Obtain a tensor of shape (batch, seq_length)
        x = x.squeeze()


        # # Pass the input through the LSTM layers and retrieve all outputs, the final hidden state
        # # and the final cell state.
        # out, (hidden, cell) = self.lstm(utterances, hidden_init)
        # 
        # # We take only the hidden state of the last layer
        # embeds_raw = self.relu(self.linear(hidden[-1]))
        # 
        # # L2-normalize it
        # embeds = embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)
        # 
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
        ## Fixme: Do you gain anything from inplace operations? Gotta check.
        ##      https://discuss.pytorch.org/t/31728
        # out = (stft ** 2).sum(dim=2)
        # out /= out.max()
        # torch.log10(out, out=out)
        # out *= 10
        # torch.max(out, out.max() - self.hparams.top_db, out=out)
        # return out

    def loss(self, y_pred, y_true):
        diff = self.spectrogram(y_pred) - self.spectrogram(y_true)
        # diff = y_pred - y_true
        l2_loss = torch.mean(diff ** 2)
        return l2_loss
