from torch import nn
import torch.nn.functional as f
import torch


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
    diff = spectrogram(y_pred, hparams) - spectrogram(y_true, hparams)
    l2_loss = torch.mean(diff ** 2)
    return l2_loss


class SimpleConvolutionalModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # Network definition
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

        x = x.squeeze(1)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()

        self.convs1 = nn.ModuleList([nn.Conv1d(in_channels=in_channels,
                                               out_channels=out_channels,
                                               padding=dilation,
                                               kernel_size=3,
                                               dilation=dilation)] * 2)
        self.convs2 = nn.ModuleList([nn.Conv1d(in_channels=out_channels,
                                               out_channels=out_channels,
                                               padding=0,
                                               kernel_size=1)] * 2)

    def forward(self, x: torch.Tensor):
        branch1_0 = torch.tanh(self.convs1[0](x))
        branch1_1 = torch.sigmoid(self.convs1[1](x))
        branch1 = branch1_0 * branch1_1
        branch2_0 = self.convs2[0](branch1)
        branch2_1 = self.convs2[1](branch1)

        residual_out = branch2_0 + x
        skip_out = branch2_1
        return residual_out, skip_out


class WavenetBasedModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.k = 16
        self.n_layers = 10

        # Network definition
        self.res_blocks = nn.ModuleList([
            ResidualBlock(
                in_channels=(1 if i == 0 else self.k),
                out_channels=self.k,
                dilation=2 ** i
            )
            for i in range(self.n_layers)
        ])
        self.conv1 = nn.Conv1d(in_channels=self.k,
                               out_channels=self.k,
                               padding=1,
                               kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=self.k,
                               out_channels=self.k,
                               padding=1,
                               kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=self.k,
                               out_channels=1,
                               padding=1,
                               kernel_size=1)

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)

        # Shape: (batch, channels, seq_length)
        residual = x
        x = torch.zeros((x.shape[0], self.k, x.shape[2])).to(x.device)
        for res_block in self.res_blocks:
            residual, skip_out = res_block(residual)
            x += skip_out

        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.conv3(x)

        x = x.squeeze(1)
        return x


class WaveUModel(nn.Module):
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
