import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import librosa
import torch


fpath =  r"D:\Users\Corentin\Music\Pure White Poison.mp3"
wav, sr = librosa.load(fpath, 44100)
wav = wav[:100000]
wav_t = torch.tensor(wav)

def spectrogram(wav, win_size, hop_size, top_db=80.0):
    stft = torch.stft(wav, win_size, hop_size, win_size, window=torch.hann_window(win_size))
    power = (stft ** 2).sum(dim=2)
    log_spec = 10. * torch.log10(power / power.max())
    torch.max(log_spec, log_spec.max() - top_db, out=log_spec)
    return log_spec

d = spectrogram(wav_t, sr // 20, sr // 80).numpy()

plt.figure(figsize=(12, 8))
librosa.display.specshow(d, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')
plt.show()
