
class HParams:
    def __init__(self):
        ## Audio parameters
        # Sample rate of the generated audio (you will have to change Timidity's parameters if you 
        # wish to change this)
        self.sample_rate = 44100
        
        # Spectrogram parameters
        self.win_size = self.sample_rate // 20    # 50ms window
        self.hop_size = self.sample_rate // 80    # 12.5ms hope size (75% overlap)
        self.n_fft = self.win_size
        self.top_db = 80.0
        
        # Samples below this value are considered to be 0
        self.silence_threshold = 1e-4
        
        
        ## MIDI & chunk parameters
        # Max duration of a midi file in seconds (longer ones are discarded)
        self.max_midi_duration = 1800
        
        # Duration of a single chunk in seconds
        self.chunk_duration = 5
        
        # Chunks will be discarded during training if there aren't enough instruments playing for
        # long enough. We take the proportion of the duration for which each instrument plays
        # individually and sum these proportions. If this value is lower than
        # <chunk_sum_prop_play_min>, the sample is discarded. This value should be between 0 
        # (all chunks are accepted, even if silent) and the number of instruments selected (a chunk
        # is only accepted if all instruments play for the entire duration of the chunk).
        self.chunk_sum_prop_play_min = 0.8

    
        ## Models parameters
        # Initial learning rate
        self.learning_rate_init = 0.01
        
        
        ## Training parameters
        # List of instruments to use for training (hand picked from the most present instruments in 
        # the dataset where very similar instruments were skipped).
        self.default_instruments = [-1, 0, 48, 33, 25, 24, 52, 73, 4, 56]
        
    def update(self, **kwargs):
        self.__dict__.update(kwargs)


hparams = HParams()
