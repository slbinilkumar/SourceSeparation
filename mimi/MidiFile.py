from scipy.io import wavfile
from typing import List
import numpy as np
import mido
import os

module_root_path = os.path.split(os.path.abspath(__file__))[0]  # mimi/
cfg_file = os.path.join(module_root_path, "soundfont", "soundfont.cfg")  # mimi/soundfont/soundfont.cfg
sf2_folder = os.path.join(module_root_path, "soundfont")  # mimi/soundfont/
default_sf2 = "8MBGMSFX.SF2"  # 8MBGMSFX.SF2


def set_soundfont(dir=None):
    if dir is None:
        with open(cfg_file, 'w') as f:
            f.write("dir {} \nsoundfont \"{}\" amp=200%".format(sf2_folder, default_sf2))
    else:
        with open(cfg_file, 'w') as f:

            folder = os.path.split(dir)[0]
            sf2 = os.path.split(dir)[1]
            f.write("dir {} \nsoundfont \"{}\" amp=200%".format(folder, sf2))
set_soundfont()


class MidiFile(mido.MidiFile):
    def __init__(self, filename=None):
        mido.MidiFile.__init__(self, filename)
        self._track_to_instrument = self._get_instrument_map()
        self.all_instruments = np.unique([i for i in self._track_to_instrument if i is not None])
                
    def _get_instrument_map(self):
        channel_to_instrument = [None] * 16
        track_to_channel = [None] * len(self.tracks)
        
        # Go through all events in the midi file. Will raise an exception if the midi is malformed.
        # It is expected that:
        #   - Only one instrument can play in a single track
        #   - A channel can be set to a single instrument once
        #   - The 10th channel (index 9) plays the Drums
        for i, track in enumerate(self.tracks):
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
                    
        # Map tracks to instruments
        return [(None if c is None else channel_to_instrument[c]) for c in track_to_channel]

    def generate_waveform(self, instruments: List[int]):
        for instrument in instruments:
            if not instrument in self.all_instruments:
                raise Exception("Instrument %d does not appear in this music")
            
        temp_mid_fpath = "temp.mid"
        temp_wav_fpath = "temp.wav"
        
        # Create a midi file with only the selected instruments and the metadata tracks
        new_mid = mido.MidiFile(type=self.type, ticks_per_beat=self.ticks_per_beat,
                                charset=self.charset)
        new_mid.tracks = [t for t, i in zip(self.tracks, self._track_to_instrument) if
                          i is None or i in instruments]
        new_mid.save(temp_mid_fpath)
        
        # Synthesize the midi to a waveform
        # --preserve-silence: if the track starts with a silence, do not skip to the first note. 
        # This keeps all instruments synced.
        # --quiet=2: do not output anything to stdout
        # -A100: set the volume to 100% (default 70%)
        # -OwM: use RIFF WAVE format, other formats have artifacts. M stands for mono.
        # -o: output filename
        os.system("timidity -c %s %s --preserve-silence --quiet=2 -A100 -OwM -o %s" % 
                  (cfg_file, temp_mid_fpath, temp_wav_fpath))
        
        sr, wav = wavfile.read(temp_wav_fpath)
        wav = wav.astype(np.float32) / 32767    # 16 bits signed to 32 bits floating point
        wav = np.clip(wav, -1, 1)   # To correct finite precision errors
        
        os.remove(temp_mid_fpath)
        os.remove(temp_wav_fpath)

        return wav, sr
