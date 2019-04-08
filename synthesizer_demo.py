import sounddevice as sd
from data_objects import Music, get_instrument_name
from params import sample_rate

# Replace with any midi file
fpath = r"E:\Datasets\Midi\FF\FF1-7 (fanmade)\FF7jenova.mid"
# fpath = r"E:\Datasets\Midi\pop_midi_dataset_ismir\cariart\full\3c4070cbe161ed8fbbb9da677bdc6cefa262335a.mid"
# fpath = r"E:\Datasets\Midi\pop_midi_dataset_ismir\cariart\full\2fb38fc9cc041622d3451f773966eeaf7e1db5d6.mid"

mid = Music(fpath)
print("Loaded track %s" % fpath)
        
# Generate an audio waveform for each track
instrument_wavs = []
for instrument_id in mid.all_instruments:
    instrument_name = get_instrument_name(instrument_id)
    print("   Generating waveform for %s" % instrument_name)
    wav = mid.generate_waveform([instrument_id])
    instrument_wavs.append(wav)

# Play all instruments seperately
print("Playing individual tracks. Some tracks may be silent at the start.")
duration = 7    # In seconds
for wav, instrument_id in zip(instrument_wavs, mid.all_instruments):
    instrument_name = get_instrument_name(instrument_id)
    print("   Now playing %d seconds of the %s track" % (duration, instrument_name))
    sd.play(wav[:sample_rate * duration], sample_rate, blocking=True)

# Play all instruments together
print("Generating waveform for all instruments")
wav_mix = mid.generate_waveform(mid.all_instruments)
print("   Now playing all instruments together")
sd.play(wav_mix, sample_rate, blocking=True)
