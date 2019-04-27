
class HParams:
    def __init__(self, **kwargs):
        self.update(**kwargs)
        
    def update(self, **kwargs):
        self.__dict__.update(kwargs)


def _define_hparams():
    ## Audio parameters
    # Sample rate of the generated audio (you will have to change Timidity's parameters if you 
    # wish to change this)
    sample_rate = 44100
    
    # Spectrogram parameters
    win_size = sample_rate // 20    # 50ms window
    hop_size = sample_rate // 80    # 12.5ms hope size (75% overlap between two consecutive windows)
    n_fft = win_size
    top_db = 80.0
    
    # Samples below this value are considered to be 0
    silence_threshold = 1e-4
    
    
    ## MIDI & chunk parameters
    # Max duration of a midi file in seconds (longer ones are discarded)
    max_midi_duration = 1800
    
    # Duration of a single chunk in seconds
    chunk_duration = 5
    
    # If more than this proportion of the waveform is equivalent between the source and the target 
    # waveform, the sample is discarded.
    chunk_equal_prop_max = 0.65
    
    # If more than this proportion of the target sample is silence, the sample is discarded.
    chunk_silence_prop_max = 0.4
    
    
    return HParams(**locals())


hparams = _define_hparams()
