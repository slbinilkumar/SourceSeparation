from pathlib import Path
from typing import List
from torch import nn
import torch
import numpy as np


def spectrogram(wav, hparams):
    stft = torch.stft(
        wav,
        n_fft=hparams.n_fft,
        hop_length=hparams.hop_size,
        win_length=hparams.win_size,
        window=torch.hann_window(hparams.win_size).cuda()
    )
    power = (stft ** 2).sum(dim=-1)
    log_spec = 10. * torch.log10(torch.clamp(power / power.max(), 1e-10))
    return torch.max(log_spec, log_spec.max() - hparams.top_db)


def spectrogram_loss(y_pred, y_true, hparams):
    # Flatten the tensors to obtain an array of wavs 
    y_pred = y_pred.reshape(-1, y_pred.shape[-1])
    y_true = y_true.reshape(-1, y_true.shape[-1])
    
    diff = spectrogram(y_pred, hparams) - spectrogram(y_true, hparams)
    l2_loss = torch.mean(diff ** 2)
    return l2_loss


def mae_loss(y_pred, y_true):
    return torch.mean(torch.abs(y_pred - y_true))


def mae_diff_loss(y_pred, y_true):
    batch_size = y_pred.shape[0]
    n = y_pred.shape[1]
    
    diff_matrix = torch.zeros(n, n)
    mask0, mask1 = np.where(np.ones((n, n)))
    
    for yi_pred, yi_true in zip(y_pred, y_true):
        sample_diff = torch.mean(torch.abs(yi_pred[mask0] - yi_true[mask1]), dim=1)
        diff_matrix[mask0, mask1] = diff_matrix[mask0, mask1] + sample_diff.cpu() 
    diff_matrix = diff_matrix / batch_size

    sim_loss = torch.mean(diff_matrix.diag())
    diff_loss = -torch.mean(diff_matrix[np.where(1 - np.eye(n))])
    
    return sim_loss, diff_loss, diff_matrix.detach().numpy()

    
def mse_loss(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)

        
class SourceSeparationModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def load(self, fpath: Path, optimizer, instruments: List[int], hparams):
        checkpoint = torch.load(fpath)
        
        # Check that the model was trained on the same set of instruments
        model_instruments = checkpoint["instruments"]
        assert np.array_equal(instruments, model_instruments), \
            "A different set of instruments was used to train this model: %s vs %s" % \
            (model_instruments, instruments)
        
        # Load the model weights
        self.load_state_dict(checkpoint["model_state"])
        
        # Load the optimizer state
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            optimizer.param_groups[0]["lr"] = hparams.learning_rate_init
        
        # Return the step at which the model was left off when training
        return checkpoint["step"]

    def forward(self, *input):
        raise NotImplemented()
    

class SimpleConvolutionalModel(SourceSeparationModel):
    def __init__(self, n_instruments, hparams):
        super().__init__()
        self.hparams = hparams
        self.n_instruments = n_instruments

        # Network definition
        inner_channels = 50
        self.conv1 = nn.Conv1d(in_channels=1,
                               out_channels=inner_channels,
                               padding=self.hparams.sample_rate // 160,
                               kernel_size=self.hparams.sample_rate // 80)
        self.conv2 = nn.Conv1d(in_channels=inner_channels,
                               out_channels=inner_channels,
                               padding=70,
                               kernel_size=141)
        self.conv3 = nn.Conv1d(in_channels=inner_channels,
                               out_channels=inner_channels,
                               padding=18,
                               kernel_size=37)
        self.conv4 = nn.Conv1d(in_channels=inner_channels,
                               out_channels=n_instruments,
                               padding=5,
                               kernel_size=11)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)

        # Shape: (batch, channels, seq_length)
        y1 = self.conv1(x)
        y1 = torch.nn.functional.relu(y1, inplace=True)
        y1 = y1 + x
        y2 = self.conv2(y1)
        y2 = torch.nn.functional.relu(y2, inplace=True)
        y2 = y2 + y1
        y3 = self.conv3(y2)
        y3 = torch.nn.functional.relu(y3, inplace=True)
        y3 = y3 + y2
        y4 = self.conv4(y3)
        
        return y4


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()

        self.conv1_1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=dilation,
            kernel_size=3,
            dilation=dilation
        )
        self.conv1_2 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=dilation,
            kernel_size=3,
            dilation=dilation
        )
        self.conv2_1 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1
        )
        self.conv2_2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1
        )

    def forward(self, x: torch.Tensor):
        branch1_1 = torch.tanh(self.conv1_1(x))
        branch1_2 = torch.sigmoid(self.conv1_2(x))
        branch1 = branch1_1 * branch1_2
        branch2_1 = self.conv2_1(branch1)
        branch2_2 = self.conv2_2(branch1)

        residual_out = branch2_1 + x
        skip_out = branch2_2
        return residual_out, skip_out
    

class ResidualStack(nn.Module):
    def __init__(self, k, n_layers, is_first):
        super().__init__()
        
        self.k = k
        self.blocks = nn.ModuleList([
            ResidualBlock(
                in_channels=(1 if (i == 0 and is_first) else k),
                out_channels=k,
                dilation=2 ** i
            )
            for i in range(n_layers)
        ])

    def forward(self, x: torch.Tensor):
        skip_out = torch.zeros((x.shape[0], self.k, x.shape[2])).to(x.device)
        for block in self.blocks:
            x, skip = block(x)
            skip_out += skip
        return x, skip_out

class WavenetBasedModel(SourceSeparationModel):
    def __init__(self, n_instruments, hparams):
        super().__init__()
        self.n_instruments = n_instruments
        self.hparams = hparams

        k = 128        # Number of inner channels in the blocks 
        n_layers = 10  # Number of blocks per stack
        r = 4          # Number of stacks in the model

        # Network definition
        self.stacks = nn.ModuleList([ResidualStack(k, n_layers, i == 0) for i in range(r)])
        self.conv1 = nn.Conv1d(in_channels=k,
                               out_channels=512,
                               padding=1,
                               kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=512,
                               out_channels=256,
                               padding=1,
                               kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=256,
                               out_channels=n_instruments,
                               kernel_size=1)

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)
        
        skip_out = None
        for stack in self.stacks:
            x, skip = stack(x)
            skip_out = skip if skip_out is None else skip_out + skip
            
        x = torch.nn.functional.relu(skip_out, inplace=True)
        x = self.conv1(x)
        x = torch.nn.functional.relu(x, inplace=True)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x, inplace=True)
        x = self.conv3(x)

        return x


class WaveUModel(SourceSeparationModel):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        down_filter_size = 25
        up_filter_size = 15
        self.num_layers = 12
        self.init_num_filters = 4

        downs = []
        for i in range(1, self.num_layers + 2):
            in_channels = 1 if i == 1 else self.init_num_filters * (i - 1)
            out_channels = self.init_num_filters * i

            downs.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=down_filter_size,
                padding=(down_filter_size - 1) // 2
            ))

        ups = [None] * self.num_layers
        for i in reversed(range(1, self.num_layers + 1)):
            in_channels = self.init_num_filters * (i + 1)
            out_channels = self.init_num_filters * i
            ups[i - 1] = nn.Conv1d(
                in_channels=in_channels * 2 - self.init_num_filters,
                out_channels=out_channels,
                kernel_size=up_filter_size,
                padding=(up_filter_size - 1) // 2
            )

        self.out_conv = nn.Conv1d(
            in_channels=5,
            out_channels=1,
            kernel_size=1,
        )

        self.down_convs = nn.ModuleList(downs)
        self.up_convs = nn.ModuleList(ups)

    def forward(self, x: torch.Tensor):
        # Obtain a tensor of shape (batch, channels, seq_length)
        x = x.unsqueeze(1)

        # Pad zeros to seq_length to the next nearest power of 2
        padding = 262144 - x.shape[2]
        x = torch.cat(
            [x, torch.zeros(x.shape[0], x.shape[1], padding).cuda()],
            dim=2
        )

        padded_input = x
        features = []

        # Down-sampling
        for i in range(self.num_layers):
            x = self.down_convs[i](x)
            x = nn.functional.leaky_relu(x)
            features.append(x)
            x = x[:, :, ::2]  # Decimation

        x = self.down_convs[self.num_layers](x)

        # Up-sampling
        for i in reversed(range(self.num_layers)):
            x = nn.functional.interpolate(x, scale_factor=2.0, mode='linear')
            f = features[i]
            x = torch.cat([x, f], dim=1)
            x = self.up_convs[i](x)
            x = nn.functional.leaky_relu(x)

        x = torch.cat([x, padded_input], dim=1)
        x = self.out_conv(x)

        # Remove initial padding
        x = x[:, :, :262144 - padding]

        # Obtain a tensor of shape (batch, seq_length)
        x = x.squeeze()

        return x

