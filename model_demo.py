from source_separation.data_objects import MidiDataset, get_instrument_id
from source_separation.params import sample_rate
import tensorflow.keras.layers as kl
import tensorflow.keras as keras
from pathlib import Path

# TODO: put as argument with argparse
dataset_root = Path(r"E:\Datasets\Midi")

batch_size = 32
chunk_duration = 5

if __name__ == "__main__":
    instruments = [get_instrument_id("Trumpet")]
    dataset = MidiDataset(
        root=dataset_root, 
        is_train=True, 
        chunk_duration=chunk_duration,
        source_instruments=instruments, 
        target_instruments=instruments,
        batch_size=batch_size,
        # Fixme: don't forget to put a larger buffer size when testing things, we put a small one 
        #  for quicker debugging only.
        shuffle_buffer= batch_size,
    )
    
    # Yields a single batch. Both x_train and y_train are of shape (batch_size, n_samples)
    x_train, y_train = next(iter(dataset))  
    
    # If you want to have a look at the data
    import sounddevice as sd
    for i in range(5):
        print("Playing chunk %d" % i)
        sd.play(x_train[i], 44100, blocking=True)

    # Example model
    model = keras.Sequential([
        kl.InputLayer(input_shape=(chunk_duration * sample_rate,)),
        kl.Dense(1024, activation='relu'),
        kl.Dense(1024, activation='relu'),
        kl.Dense(chunk_duration * sample_rate, activation='tanh')
    ])

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['accuracy']
    )

    # TODO Compute appropriate value for the # of steps
    model.fit(x_train, y_train, epochs=2, steps_per_epoch=10)  
