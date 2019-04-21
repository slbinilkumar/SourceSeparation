import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """
    Parse the CLI arguments of a module
    :return: a Namespace containing the arguments
    """

    # RawTextHelpFormatter avoids clipping the help messages
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("dataset_root",
                        help="Path to a directory containing the 'pop_midi_dataset_ismir' dataset and the index files")
    parser.add_argument("source_instruments",
                        help="List of names of instruments. "
                             "For a complete list: python -m source_separation.data_objects.midi_instruments")
    parser.add_argument("target_instruments", help="Same as source_instruments")

    parser.add_argument("--batch_size", default=32)
    parser.add_argument("--chunk_duration", default=5)
    parser.add_argument("--sample_rate", default=44100)
    parser.add_argument("--train_steps", default=20000)
    parser.add_argument("--validate_steps", default=1000)

    args = parser.parse_args()

    # Converts the string as a Path from pathlib
    args.dataset_root = Path(args.dataset_root)

    return args
