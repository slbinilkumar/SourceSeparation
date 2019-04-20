from source_separation.data_objects import MidiDataset, get_instrument_id
from source_separation.params import sample_rate
import tensorflow.keras.layers as kl
import tensorflow.keras as keras
# Fixme: remove these two imports later
import tensorflow.python.keras.api._v2.keras.layers as kl
from tensorflow.python.keras.api._v2 import keras
from pathlib import Path

# TODO: put as argument with argparse
dataset_root = Path(r"E:\Datasets\Midi")

batch_size = 32
chunk_duration = 5
validate_every = 1000
train_steps = 20000

if __name__ == "__main__":
    source_instruments = "AcousticGrandPiano,Trumpet"
    target_instruments = "Trumpet"
    get_instruments_id = lambda l: list(map(get_instrument_id, l.split(",")))
    
    dataset = MidiDataset(
        root=dataset_root, 
        is_train=True, 
        chunk_duration=chunk_duration,
        source_instruments=get_instruments_id(source_instruments), 
        target_instruments=get_instruments_id(target_instruments),
        batch_size=batch_size,
        # Fixme: don't forget to put a larger buffer size later, we put a small one 
        #  for quicker debugging only.
        shuffle_buffer= batch_size,
    )
    
    # # If you want to have a look at the data
    # import sounddevice as sd
    # x_train, y_train = next(iter(dataset))
    # for i in range(5):
    #     print("Playing chunk %d" % i)
    #     sd.play(x_train[i], 44100, blocking=True)
    
    # Example model
    identity_layer = kl.Conv1D(1, 1, input_shape=(chunk_duration * sample_rate, 1), use_bias=False)
    
    model = keras.Sequential([
        kl.Reshape((chunk_duration * sample_rate, 1)),
        kl.Conv1D(1, 1, input_shape=(chunk_duration * sample_rate, 1), use_bias=False),
        kl.Reshape((chunk_duration * sample_rate,)),
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mse'],
    )
    
    model.fit_generator(
        dataset, 
        steps_per_epoch=validate_every,
        epochs=train_steps // validate_every,
        shuffle=False,
    )
