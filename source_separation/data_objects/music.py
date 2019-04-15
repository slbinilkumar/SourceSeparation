from source_separation.params import sample_rate, project_root
from scipy.io import wavfile
from typing import List
from mido import MidiFile
import numpy as np
import os

# Prepare the synthesizer config
synthesizer_root = os.path.join(project_root, "synthesizer")
config_fpath = os.path.join(synthesizer_root, "timidity.cfg")
with open(config_fpath, 'w') as config:
    config.write("dir \"%s\"\nsoundfont \"%s\"" % (synthesizer_root, "soundfont.sf2"))

class Music:
    def __init__(self, fpath=None):
        self.mid = MidiFile(fpath)
        self._track_to_instrument = self._get_instrument_map()
        self.all_instruments = np.unique([i for i in self._track_to_instrument if i is not None])
        self.wav_length = int(np.ceil(self.mid.length * sample_rate))
                
    def _get_instrument_map(self):
        channel_to_instrument = [None] * 16
        track_to_channel = [None] * len(self.mid.tracks)
        
        # Go through all events in the midi file. Will raise an exception if the midi is malformed.
        # It is expected that:
        #   - Only one instrument can play in a single track
        #   - A channel can be set to a single instrument once
        #   - The 10th channel (index 9) plays the Drums
        #   - Any channel whose instrument isn't specified defaults to piano (id: 0)
        for i, track in enumerate(self.mid.tracks):
            for event in track:
                # Any track that plays a note is an instrument track
                if event.type == "note_on":
                    assert track_to_channel[i] is None or track_to_channel[i] == event.channel
                    track_to_channel[i] = event.channel
                    
                # Register instruments being set
                if event.type == "program_change":
                    assert channel_to_instrument[event.channel] is None or \
                           channel_to_instrument[event.channel] == event.program
                    channel_to_instrument[event.channel] = event.program
        channel_to_instrument[9] = -1   # Special case of the drums
        
        # Replace missing instruments by the piano
        channel_to_instrument = [c if c is not None else 0 for c in channel_to_instrument]
                    
        # Map tracks to instruments
        return [(None if c is None else channel_to_instrument[c]) for c in track_to_channel]
    
    def generate_waveform(self, instruments: List[int], synchronized=True):
        """
        Synthesizes a waveform from the midi file with only a subset of instruments playing.
        
        :param instruments: a list of instrument IDs to be included in the waveform.
        :param synchronized: if False, the audio will be generated starting at the first note the 
        instruments play, and will end on the last note. If True, the audio will start at the 
        beginning of the music (even if silent) and will end at the end of the music. All audio 
        waveforms from the same Music generated with synchronize=True will thus be of the exact 
        same length.
        :return: the waveform as a float32 numpy array of shape (n_samples,) 
        """
        for instrument in instruments:
            if not instrument in self.all_instruments:
                raise Exception("Instrument %d does not appear in this music")
            
        # TODO: adapt this for multithreading and put in the OS' temp directory
        temp_mid_fpath = "temp.mid"
        temp_wav_fpath = "temp.wav"
        
        # Create a midi file with only the selected instruments and the metadata tracks
        new_mid = MidiFile(type=self.mid.type, ticks_per_beat=self.mid.ticks_per_beat,
                           charset=self.mid.charset)
        new_mid.tracks = [t for t, i in zip(self.mid.tracks, self._track_to_instrument) if
                          i is None or i in instruments]
        new_mid.save(temp_mid_fpath)
        
        # Synthesize the midi to a waveform
        # -c: config file, contains the path to the soundfont.
        # --quiet=2: do not output anything to stdout.
        # -A100: set the volume to 100% (default 70%).
        # -OwM: use RIFF WAVE format, other formats have artifacts. M stands for mono.
        # --preserve-silence: if the track starts with a silence, do not skip to the first note. 
        options = f"-c {config_fpath} --quiet=2 -A100 -OwM"
        options += " --preserve-silence" if synchronized else ""
        timidity_fpath = os.path.join("synthesizer", "timidity")    # Path to the executable
        os.system(f"{timidity_fpath} {temp_mid_fpath} {options} -o {temp_wav_fpath}")
        os.remove(temp_mid_fpath)

        # Retrieve the waveform
        try:
            sr, wav = wavfile.read(temp_wav_fpath)
        except FileNotFoundError:
            raise Exception("Failed to generate a waveform. Make sure that the Timidity "
                            "executable in synthesizer/ is operational, and that timidity.cfg "
                            "points to the soundfont in the same directory.")
        os.remove(temp_wav_fpath)
        wav = wav.astype(np.float32) / 32767    # 16 bits signed to 32 bits floating point
        wav = np.clip(wav, -1, 1)   # To correct finite precision errors
        
        # Pad or trim the waveform to the length of the track
        if synchronized and len(wav) > self.wav_length:
            wav = wav[:self.wav_length]
        if synchronized and len(wav) < self.wav_length:
            wav = np.pad(wav, (0, self.wav_length - len(wav)), "constant")

        return wav
