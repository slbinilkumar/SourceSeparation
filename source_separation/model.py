from torch import nn
import numpy as np
import torch


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Network defition
        self.conv1 = nn.Conv1d(in_channels=1,
                               out_channels=1,
                               kernel_size=1)
        # self.lstm = nn.LSTM(input_size=mel_n_channels,
        #                     hidden_size=model_hidden_size,
        #                     num_layers=model_num_layers,
        #                     batch_first=True).to(device)
        # self.relu = torch.nn.ReLU().to(device)
    
    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)
        x = self.conv1(x)
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
    
    def loss(self, y_pred, y_true):
        pass
    
# 
# def spectrogram(wav, win_size, hop_size, amin=1e-6, top_db=80.0):
#     s = stft(wav, win_size, hop_size, win_size)
#     return tf.abs(s)
#     power = tf.square(tf.abs(s))
#     a = tf.maximum(amin, power)
#     return log_spec
#     log_spec = tf.math.log(a)
#     # log_spec = tf.math.log(tf.maximum(amin, power / tf.reduce_max(power)))
#     log_spec = 10.0 * log_spec / tf.math.log(tf.constant(10.))
#     return tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)
# 
# def loss_fn(y_true, y_pred):
#     spec = lambda wav: spectrogram(wav, 44100 // 20, 44100 // 80)
#     loss = tf.reduce_mean(tf.abs(tf.map_fn(spec, y_true) - tf.map_fn(spec, y_pred)))
#     return loss
