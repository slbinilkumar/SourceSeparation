from source_separation.hparams import hparams
from source_separation.generate import generate
from pathlib import Path
import argparse


if __name__ == '__main__':
    class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass
    
    # Parse the arguments from cli
    default_instruments = ",".join(map(str, hparams.default_instruments))
    parser = argparse.ArgumentParser(description="Generates the output of a trained model.",
                                     formatter_class=MyFormatter)
    parser.add_argument("run_name",
                        help="Name of the trained model.")
    parser.add_argument("midi_fpath",
                        help="Path to a midi file to generate.")
    parser.add_argument("-i", "--instruments", default=default_instruments, type=str,
                        help="Comma-separated list of instruments ids. For a complete list of "
                             "available instruments: python -m "
                             "source_separation.data_objects.midi_instruments")
    parser.add_argument("-o", "--out_dir", default="samples", type=str,
                        help="Output directory")
    parser.add_argument("--music_duration", default=20, type=float,
                        help="Duration of the music, in seconds")
    parser.add_argument("--chunk_duration", default=20, type=float,
                        help="Duration of the chunks, in seconds. Increase this as much as "
                             "possible depending on your GPU's memory.")
    
    # Format the arguments
    args = parser.parse_args()
    args.midi_fpath = Path(args.midi_fpath)
    args.out_dir = Path(args.out_dir)
    args.instruments = list(map(int, args.instruments.split(",")))

    generate(args, hparams)
    