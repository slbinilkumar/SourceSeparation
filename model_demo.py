from source_separation.data_objects import MidiDataset, get_instrument_id
from source_separation.params import sample_rate
import tensorflow.keras.layers as kl
import tensorflow.keras as keras
from pathlib import Path

# TODO: put as argument with argparse
dataset_root = Path(r"E:\Datasets\Midi")

batch_size = 32
sample_duration = 5

if __name__ == "__main__":
    instruments = [get_instrument_id("Trumpet")]
    dataset = MidiDataset(
        dataset_root, 
        is_train=True, 
        sample_duration=5,
        source_instruments=instruments, 
        target_instruments=instruments,
    )
    # Fixme: don't forget to put a larger buffer size when testing things, we put a small one for
    #   quicker debugging only.
    dataset = dataset.generate_dataset(batch_size, shuffle_buffer=batch_size)   
    # Yields a single batch. Both x_train and y_train are of shape (batch_size, n_samples)
    x_train, y_train = next(iter(dataset))  

    # Example model
    model = keras.Sequential([
        kl.InputLayer(input_shape=(sample_duration * sample_rate,)),
        kl.Dense(1024, activation='relu'),
        kl.Dense(1024, activation='relu'),
        kl.Dense(sample_duration * sample_rate, activation='tanh')
    ])

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['accuracy']
    )

    # TODO Compute appropriate value for the # of steps
    model.fit(x_train, y_train, epochs=2, steps_per_epoch=10)  
