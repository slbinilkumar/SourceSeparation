from mimi import MidiFile
from mimi.instrument import get_instrument_name
import sounddevice as sd
import numpy as np

fpath = r"E:\Datasets\Midi\FF\FF1-7 (fanmade)\FF7jenova.mid"
# fpath = r"E:\Datasets\Midi\pop_midi_dataset_ismir\cariart\full\3c4070cbe161ed8fbbb9da677bdc6cefa262335a.mid"
# fpath = r"E:\Datasets\Midi\pop_midi_dataset_ismir\cariart\full\2fb38fc9cc041622d3451f773966eeaf7e1db5d6.mid"
mid = MidiFile(fpath)
print("Loaded track %s" % fpath)
        
# Generate an audio waveform for each track
instrument_wavs = []
sample_rate = 44100
for instrument_id in mid.all_instruments:
    instrument_name = get_instrument_name(instrument_id)
    print("   Generating waveform for %s" % instrument_name)
    wav, gen_sample_rate = mid.generate_waveform([instrument_id])
    assert gen_sample_rate == sample_rate # All audios should have that same sample rate.
    instrument_wavs.append(wav)

# Pad the end of the audio tracks so that they have the same duration
max_len = max(len(w) for w in instrument_wavs)
instrument_wavs = [np.pad(w, (0, max_len - len(w)), "constant") for w in instrument_wavs]

# Play all instruments seperately
print("Playing individual tracks. Some tracks may be silent at the start.")
duration = 7    # In seconds
for wav, instrument_id in zip(instrument_wavs, mid.all_instruments):
    instrument_name = get_instrument_name(instrument_id)
    print("   Now playing %d seconds of the %s track" % (duration, instrument_name))
    sd.play(wav[:sample_rate * duration], sample_rate, blocking=True)

# Play all instruments together
print("Generating waveform for all instruments")
wav_mix, sample_rate = mid.generate_waveform(mid.all_instruments)
print("   Now playing all instruments together")
sd.play(wav_mix, sample_rate, blocking=True)
