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
        n_threads=1,
    )

    # # If you want to have a look at the data
    # import sounddevice as sd
    # x_train, y_train = next(iter(dataset))
    # dataset._debug_accept_rate()
    # for i in range(15):
    #     print("Playing chunk %d" % i)
    #     sd.play(x_train[i], 44100, blocking=True)

    model = keras.Sequential([
        kl.Reshape((args.chunk_duration * args.sample_rate, 1)),
        # kl.Conv1D(
        #     filters=30,
        #     # kernel_size=1,
        #     kernel_size=args.sample_rate // 10,     # Window of 1/10th of a second 
        # ),
        kl.Conv1D(
            filters=1,
            kernel_size=1,
        ),
        kl.Reshape((args.chunk_duration * args.sample_rate,)),
    ])

    # TODO: this will be moved later, just testing here
    import tensorflow as tf
    def spectrogram(wav, win_size, hop_size, amin=1e-10, top_db=80.0):
        stft = tf.transpose(tf.signal.stft(wav, win_size, hop_size, win_size), (1, 0))
        power = tf.square(tf.abs(stft))
        log_spec = tf.math.log(tf.maximum(amin, power / tf.reduce_max(power)))
        log_spec = 10.0 * log_spec / tf.math.log(tf.constant(10.))
        return tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)

    def loss(y_true, y_pred):
        spec = lambda wav: spectrogram(wav, 44100 // 20, 44100 // 80)
        loss = tf.reduce_mean(tf.square(tf.map_fn(spec, y_true) - tf.map_fn(spec, y_pred)))
        return loss

    model.compile(
        optimizer='adam',
        loss=loss,
    )
    
    model.fit_generator(
        dataset,
        steps_per_epoch=args.validate_steps,
        epochs=args.train_steps // args.validate_steps,
        shuffle=False,
    )
