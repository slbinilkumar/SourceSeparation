import tensorflow.keras as keras
import tensorflow.keras.layers as kl

from data_objects.dataset import generate_dataset
from params import sample_rate

BATCH_SIZE = 32
SAMPLE_DURATION = 5
PATH_INDEX = 'index.txt'

if __name__ == "__main__":
    dataset = generate_dataset('index.txt', BATCH_SIZE, SAMPLE_DURATION, shuffle_buffer=2048)
    x_train, ids = next(iter(dataset))  # Yields a tuple (waveform, instrument ID)

    # Example model
    model = keras.Sequential([
        kl.InputLayer(input_shape=(SAMPLE_DURATION * sample_rate,)),
        kl.Dense(1024, activation='relu'),
        kl.Dense(1024, activation='relu'),
        kl.Dense(SAMPLE_DURATION * sample_rate, activation='tanh')
    ])

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['accuracy']
    )

    model.fit(x_train, x_train, epochs=2, steps_per_epoch=10)  # TODO Compute appropriate value for the # of steps
