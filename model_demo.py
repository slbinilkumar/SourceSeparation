from source_separation.data_objects import MidiDataset, get_instrument_id
import tensorflow.keras.layers as kl
import tensorflow.keras as keras
# Fixme: remove these two imports later
import tensorflow.python.keras.api._v2.keras.layers as kl
from tensorflow.python.keras.api._v2 import keras

from source_separation.parser import parse_args

if __name__ == "__main__":
    args = parse_args()

    source_instruments = args.source_instruments
    target_instruments = args.target_instruments
    get_instruments_id = lambda l: list(map(get_instrument_id, l.split(",")))

    dataset = MidiDataset(
        root=args.dataset_root,
        is_train=True,
        chunk_duration=args.chunk_duration,
        source_instruments=get_instruments_id(source_instruments),
        target_instruments=get_instruments_id(target_instruments),
        batch_size=args.batch_size,
        sample_rate=args.sample_rate,
        # Fixme: don't forget to put a larger buffer size later, we put a small one 
        #  for quicker debugging only.
        shuffle_buffer=args.batch_size,
    )

    # # If you want to have a look at the data
    # import sounddevice as sd
    # x_train, y_train = next(iter(dataset))
    # for i in range(5):
    #     print("Playing chunk %d" % i)
    #     sd.play(x_train[i], 44100, blocking=True)

    # Example model
    identity_layer = kl.Conv1D(1, 1, input_shape=(args.chunk_duration * args.sample_rate, 1), use_bias=False)

    model = keras.Sequential([
        kl.Reshape((args.chunk_duration * args.sample_rate, 1)),
        kl.Conv1D(1, 1, input_shape=(args.chunk_duration * args.sample_rate, 1), use_bias=False),
        kl.Reshape((args.chunk_duration * args.sample_rate,)),
    ])

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mse'],
    )

    model.fit_generator(
        dataset,
        steps_per_epoch=args.validate_steps,
        epochs=args.train_steps // args.validate_steps,
        shuffle=False,
    )
