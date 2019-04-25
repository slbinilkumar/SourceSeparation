from source_separation.data_objects import MidiDataset, get_instrument_id
from source_separation.parser import parse_args

# Temporary, until we make our own hparams
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
hparams = Namespace(
    sample_rate=44100,
    
    # Max duration of a midi file in seconds (longer ones are discarded)
    max_midi_duration=1800,     
    
    # Duration of a single chunk in seconds
    chunk_duration=5,
    
    # Samples below this value are considered to be 0
    silence_threshold = 1e-4,        
    
    # If more than this proportion of the waveform is equivalent between the source and the target 
    # waveform, the sample is discarded.
    chunk_equal_prop_max = 0.65, 
    
    # If more than this proportion of the target sample is silence, the sample is discarded.
    chunk_silence_prop_max = 0.4,
)


if __name__ == "__main__":
    args = parse_args()
    source_instruments = args.source_instruments
    target_instruments = args.target_instruments
    get_instruments_id = lambda l: list(map(get_instrument_id, l.split(",")))
    
    dataset = MidiDataset(
        root=args.dataset_root,
        is_train=True,
        hparams=hparams,
    )
    
    dataloader = dataset.generate(
        source_instruments=get_instruments_id(source_instruments),
        target_instruments=get_instruments_id(target_instruments),
        batch_size=args.batch_size,
        n_threads=4,
        music_buffer_size=8, # Put a low value for debugging, a higher one for training.
    )
    
    # If you want to test the speed of the dataloader
    from time import sleep
    for i, batch in enumerate(dataloader, 1):
        print("Simulating training step %d on batch of shape %s" % (i, str(batch.shape)))
        sleep(1)
    quit()

    # If you want to have a look at the data
    import sounddevice as sd
    x_train, y_train = next(dataloader)
    for i in range(15):
        print("Playing chunk %d" % i)
        sd.play(x_train[i], 44100, blocking=True)
    quit()
    
    
    # Todo: adapt all this to pytorch
    model = keras.Sequential([
        kl.Reshape((args.chunk_duration * args.sample_rate, 1)),
        kl.Conv1D(
            filters=5,
            kernel_size=args.sample_rate // 10,     # Window of 1/10th of a second 
            padding="same"
        ),
        kl.Conv1D(
            filters=1,
            kernel_size=1,
        ),
        kl.Reshape((args.chunk_duration * args.sample_rate,)),
    ])

    # TODO: this will be moved later, just testing here
    
    import tensorflow as tf
    from tensorflow.python.framework import ops
    from tensorflow.python.ops.signal import fft_ops
    from tensorflow.python.ops.signal import shape_ops
    from tensorflow.python.ops.signal import window_ops
    from tensorflow.python.ops.signal.spectral_ops import _enclosing_power_of_two
    def stft(signals, frame_length, frame_step, fft_length=None, window_fn=window_ops.hann_window,
             pad_end=False, name=None):
        with ops.name_scope(name, 'stft', [signals, frame_length,
                                           frame_step]):
            signals = ops.convert_to_tensor(signals, name='signals')
            signals.shape.with_rank_at_least(1)
            frame_length = ops.convert_to_tensor(frame_length, name='frame_length')
            frame_length.shape.assert_has_rank(0)
            frame_step = ops.convert_to_tensor(frame_step, name='frame_step')
            frame_step.shape.assert_has_rank(0)
        
            if fft_length is None:
                fft_length = _enclosing_power_of_two(frame_length)
            else:
                fft_length = ops.convert_to_tensor(fft_length, name='fft_length')
        
            framed_signals = shape_ops.frame(
                signals, frame_length, frame_step, pad_end=pad_end)
        
            # Optionally window the framed signals.
            if window_fn is not None:
                window = window_fn(frame_length, dtype=framed_signals.dtype)
                framed_signals *= window
        
            # fft_ops.rfft produces the (fft_length/2 + 1) unique components of the
            # FFT of the real windowed signals in framed_signals.
            return fft_ops.rfft(framed_signals, [fft_length])
    
    def spectrogram(wav, win_size, hop_size, amin=1e-6, top_db=80.0):
        s = stft(wav, win_size, hop_size, win_size)
        return tf.abs(s)
        power = tf.square(tf.abs(s))
        a = tf.maximum(amin, power)
        return log_spec
        log_spec = tf.math.log(a)
        # log_spec = tf.math.log(tf.maximum(amin, power / tf.reduce_max(power)))
        log_spec = 10.0 * log_spec / tf.math.log(tf.constant(10.))
        return tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)

    def loss_fn(y_true, y_pred):
        spec = lambda wav: spectrogram(wav, 44100 // 20, 44100 // 80)
        loss = tf.reduce_mean(tf.abs(tf.map_fn(spec, y_true) - tf.map_fn(spec, y_pred)))
        return loss

    model.compile(
        optimizer='adam',
        loss=loss_fn,
    )

    for step, (xi_train, yi_train) in enumerate(iter(dataset)):
        print("Step %d" % step, end="")
        loss = model.train_on_batch(xi_train, yi_train)
        print("   Loss: %.4f" % loss)
        print("Waiting for next batch...")
    
    # model.fit_generator(
    #     dataset,
    #     steps_per_epoch=args.validate_steps,
    #     epochs=args.train_steps // args.validate_steps,
    #     shuffle=False,
    # )
