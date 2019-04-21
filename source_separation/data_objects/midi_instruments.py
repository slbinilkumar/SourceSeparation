_instrument_to_id = {
    'Drums': -1,  # This one isn't the midi standard, just a choice we're going with.
    # Piano
    'AcousticGrandPiano': 0,
    'BrightAcousticPiano': 1,
    'ElectricGrandPiano': 2,
    'HonkyTonkPiano': 3,
    'ElectricPiano1': 4,
    'ElectricPiano2': 5,
    'Harpsichord': 6,
    'Clavinet': 7,
    # Chromatic percussion
    'Celesta': 8,
    'Glockenspiel': 9,
    'MusicBox': 10,
    'Vibraphone': 11,
    'Marimba': 12,
    'Xylophone': 13,
    'TubularBells': 14,
    'Dulcimer': 15,
    # Organ
    'DrawbarOrgan': 16,
    'PercussiveOrgan': 17,
    'RockOrgan': 18,
    'ChurchOrgan': 19,
    'ReedOrgan': 20,
    'Accordion': 21,
    'Harmonica': 22,
    'TangoAccordion': 23,
    # Guitar
    'AcousticGuitarNylon': 24,
    'AcousticGuitarSteel': 25,
    'ElectricGuitarJazz': 26,
    'ElectricGuitarClean': 27,
    'ElectricGuitarMuted': 28,
    'OverdrivenGuitar': 29,
    'DistortionGuitar': 30,
    'GuitarHarmonics': 31,
    # Bass
    'AcousticBass': 32,
    'ElectricBassFinger': 33,
    'ElectricBassPick': 34,
    'FretlessBass': 35,
    'SlapBass1': 36,
    'SlapBass2': 37,
    'SynthBass1': 38,
    'SynthBass2': 39,
    # Strings
    'Violin': 40,
    'Viola': 41,
    'Cello': 42,
    'Contrabass': 43,
    'TremoloStrings': 44,
    'PizzicatoStrings': 45,
    'OrchestralHarp': 46,
    'Timpani': 47,
    # Ensemble
    'StringEnsemble1': 48,
    'StringEnsemble2': 49,
    'SynthStrings1': 50,
    'SynthStrings2': 51,
    'ChoirAahs': 52,
    'VoiceOohs': 53,
    'SynthChoir': 54,
    'OrchestraHit': 55,
    # Brass
    'Trumpet': 56,
    'Trombone': 57,
    'Tuba': 58,
    'MutedTrumpet': 59,
    'FrenchHorn': 60,
    'BrassSection': 61,
    'SynthBrass1': 62,
    'SynthBrass2': 63,
    # Reed
    'SopranoSax': 64,
    'AltoSax': 65,
    'TenorSax': 66,
    'BaritoneSax': 67,
    'Oboe': 68,
    'EnglishHorn': 69,
    'Bassoon': 70,
    'Clarinet': 71,
    # Pipe
    'Piccolo': 72,
    'Flute': 73,
    'Recorder': 74,
    'PanFlute': 75,
    'Blownbottle': 76,
    'Shakuhachi': 77,
    'Whistle': 78,
    'Ocarina': 79,
    # Synth lead
    'Lead1Square': 80,
    'Lead2Sawtooth': 81,
    'Lead3Calliope': 82,
    'Lead4Chiff': 83,
    'Lead5Charang': 84,
    'Lead6Voice': 85,
    'Lead7Fifths': 86,
    'Lead8BassLead': 87,
    # Synth pad
    'Pad1Newage': 88,
    'Pad2Warm': 89,
    'Pad3Polysynth': 90,
    'Pad4Choir': 91,
    'Pad5Bowed': 92,
    'Pad6Metallic': 93,
    'Pad7Halo': 94,
    'Pad8Xweep': 95,
    # Synth effects
    'FX1Rain': 96,
    'FX2Soundtrack': 97,
    'FX3Crystal': 98,
    'FX4Atmosphere': 99,
    'FX5Brightness': 100,
    'FX6Goblins': 101,
    'FX7Echoes': 102,
    'FX8Sci_fi': 103,
    # Ethnic
    'Sitar': 104,
    'Banjo': 105,
    'Shamisen': 106,
    'Koto': 107,
    'Kalimba': 108,
    'Bagpipe': 109,
    'Fiddle': 110,
    'Shanai': 111,
    # Percussive
    'TinkleBell': 112,
    'Agogo': 113,
    'SteelDrums': 114,
    'Woodblock': 115,
    'TaikoDrum': 116,
    'MelodicTom': 117,
    'SynthDrum': 118,
    'ReverseCymbal': 119,
    # Soundeffects
    'GuitarFretNoise': 120,
    'BreathNoise': 121,
    'Seashore': 122,
    'BirdTweet': 123,
    'TelephoneRing': 124,
    'Helicopter': 125,
    'Applause': 126,
    'Gunshot': 127
}
_inverse_index = [0] * len(_instrument_to_id)
for instrument_name, instrument_id in _instrument_to_id.items():
    _inverse_index[instrument_id] = instrument_name


def get_instrument_name(instrument_id):
    return _inverse_index[instrument_id]


def get_instrument_id(instrument_name):
    return _instrument_to_id[instrument_name]


if __name__ == "__main__":
    import pprint

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(_instrument_to_id)
