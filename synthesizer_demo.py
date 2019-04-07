from mimi import MidiFile
from mimi.instrument import get_instrument_name
import sounddevice as sd
import numpy as np


fpath = r"E:\Datasets\Midi\FF\FF1-7 (fanmade)\FF1airsh.mid"
# fpath = r"E:\Datasets\Midi\pop_midi_dataset_ismir\cariart\full\3c4070cbe161ed8fbbb9da677bdc6cefa262335a.mid"
mid = MidiFile(fpath)
print("Loaded track %s" % fpath)

# Separate the tracks that have instruments from those that don't
metadata_tracks = []
instrument_tracks = []
instrument_names = []

def get_track_instrument(track):
    """
    Retrieves the instrument ID from a track (see mimi.instrument). Returns None if the track has no
    instrument. 
    
    Note: there are (apparently) 3 ways to code an instrument:
       - With the message "program_change" followed by the ID of the instrument.
       - By setting the track's channel to 9 (0-indexed -> 10th channel), in which case 
       regardless of any program change the track will be drums. 
       - Consecutive tracks set to piano should represent a single instrument. NOTE: I haven't 
       implemented this, I have yet to find a midi file that does that.
    """
    for msg in track:
        if msg.type == "channel_prefix" and msg.channel == 9:
            return -1   # Special case of the drums
        if msg.type == "program_change":
            return msg.program
    return None

for i, track in enumerate(mid.tracks):
    instrument_id = get_track_instrument(track)
    if instrument_id is None:
        metadata_tracks.append(track)
    else:
        instrument_tracks.append(track)
        instrument_names.append(get_instrument_name(instrument_id))
print("Found %d instrument tracks and %d metadata tracks." % 
      (len(instrument_tracks), len(metadata_tracks)))
        
# Generate an audio waveform for each track
instrument_wavs = []
mid.tracks = metadata_tracks    # Remove all instrument tracks from the midi
sample_rate = 44100
for track, instrument_name in zip(instrument_tracks, instrument_names):
    print("   Generating waveform for %s" % instrument_name)
    mid.tracks.append(track)    # Add a single instrument
    wav, gen_sample_rate = mid.generate_waveform()
    assert gen_sample_rate == sample_rate # All audios should have that same sample rate.
    instrument_wavs.append(wav)
    del mid.tracks[-1]          # Remove the last instrument added
    
# Pad the end of the audio tracks so that they have the same duration
max_len = max(len(w) for w in instrument_wavs)
instrument_wavs = [np.pad(w, (0, max_len - len(w)), "constant") for w in instrument_wavs]

# Play all instruments seperately
print("Playing individual tracks. Some tracks may be silent.")
duration = 7    # In seconds
for wav, instrument_name in zip(instrument_wavs, instrument_names):
    print("   Now playing %d seconds of the %s track" % (duration, instrument_name))
    sd.play(wav[:sample_rate * duration], sample_rate, blocking=True)

# Mix instruments together
print("Mixing instruments together.")
wav_mix = np.sum(instrument_wavs, axis=0)
wav_mix /= np.max(np.abs(wav_mix))
print("   Now playing the entire track")
sd.play(wav_mix, sample_rate, blocking=True)
