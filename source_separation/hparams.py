
class HParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
    # Todo: a method to update the args with newer values if we want to parse hparams from the 
    #   command line?

hparams = HParams(
    sample_rate=44100,
    
    # Max duration of a midi file in seconds (longer ones are discarded)
    max_midi_duration=1800,
    
    # Duration of a single chunk in seconds
    chunk_duration=5,
    
    # Samples below this value are considered to be 0
    silence_threshold=1e-4,
    
    # If more than this proportion of the waveform is equivalent between the source and the target 
    # waveform, the sample is discarded.
    chunk_equal_prop_max=0.65,
    
    # If more than this proportion of the target sample is silence, the sample is discarded.
    chunk_silence_prop_max=0.4,
)
