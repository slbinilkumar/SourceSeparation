import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa.display
import librosa

fpath =  r"D:\Users\Corentin\Music\Pure White Poison.mp3"
wav, sr = librosa.load(fpath, 44100)
wav = wav[:100000]
wav_t = tf.convert_to_tensor(wav)


def spectrogram(wav, win_size, hop_size, amin=1e-10, top_db=80.0):
    stft = tf.transpose(tf.signal.stft(wav, win_size, hop_size, win_size), (1, 0))
    power = tf.square(tf.abs(stft))
    log_spec = tf.math.log(tf.maximum(amin, power / tf.reduce_max(power)))
    log_spec = 10.0 * log_spec / tf.math.log(tf.constant(10.))
    return tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)


d = spectrogram(wav, sr // 20, sr // 80)

plt.figure(figsize=(12, 8))
librosa.display.specshow(d.numpy(), y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')
plt.show()
