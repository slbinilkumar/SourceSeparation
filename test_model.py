from source_separation.data_objects import get_instrument_id
from source_separation.hparams import hparams
from source_separation.test import test
from pathlib import Path
import argparse


if __name__ == '__main__':
    class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass
    
    # Parse the arguments from cli
    default_instruments = ",".join(map(str, hparams.default_instruments))
    parser = argparse.ArgumentParser(description="Tests the source separation model.",
                                     formatter_class=MyFormatter)
    parser.add_argument("run_name",
                        help="Name of the trained model.")
    parser.add_argument("dataset_root",
                        help="Path to a directory containing the 'pop_midi_dataset_ismir' dataset "
                             "and the index files")
    parser.add_argument("-i", "--instruments", default=default_instruments, type=str,
                        help="Comma-separated list of instruments ids. For a complete list of "
                             "available instruments: python -m "
                             "source_separation.data_objects.midi_instruments")
    parser.add_argument("-d", "--chunk_duration", default=5, type=int,
                        help="Duration of the chunks to be played, in seconds")
    parser.add_argument("--sample_rate", default=44100, type=int)
    
    # Format the arguments
    args = parser.parse_args()
    args.dataset_root = Path(args.dataset_root)
    args.instruments = list(map(int, args.instruments.split(",")))

    test(args, hparams)