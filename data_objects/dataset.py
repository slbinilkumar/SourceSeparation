import numpy as np
import tensorflow as tf

from params import sample_rate
from data_objects import Music


def preprocess_waveform(waveform, duration):
    padding = (duration * sample_rate) - len(waveform) % (duration * sample_rate)
    waveform = np.append(waveform, np.zeros(padding))
    return np.vstack(np.split(waveform, len(waveform) / (duration * sample_rate)))


def generate_waveform(index_line, sample_duration):
    path, instruments = index_line.numpy().decode().split(sep=':')
    instruments = list(map(int, instruments.strip().split(sep=',')))

    m = Music(path.replace('"', ''))
    ids = np.array([])
    samples = []
    for instrument in instruments:
        wav = m.generate_waveform([instrument])
        wav = preprocess_waveform(wav, sample_duration)
        samples.append(wav)
        ids = np.append(ids, np.full(wav.shape[0], fill_value=instrument))

    return np.vstack(samples), ids


def generate_dataset(index_filepath, batch_size, sample_duration=5, shuffle_buffer=1):
    """
    Generates a tf.data.Dataset using midi files located in the index file
    :param index_filepath: path to the index file
    :param batch_size: size of the batch
    :param sample_duration: duration of one sample in seconds
    :param shuffle_buffer: size of the buffer used for randomization
    :return: an instance of a tf.data.Dataset
    """
    with open(index_filepath) as file:
        indexes = file.readlines()

    dataset = tf.data.Dataset.from_tensor_slices(indexes)
    dataset = dataset.map(lambda filename: tf.py_function(
        generate_waveform,
        [filename, sample_duration],
        [tf.float32, tf.int32]
    ))

    dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))

    # A bigger buffer gives a better randomization but takes more memory
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)

    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset
