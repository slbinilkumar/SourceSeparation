from source_separation.data_objects import get_instrument_id
from source_separation.hparams import hparams
from source_separation.train import train
from pathlib import Path
import argparse


if __name__ == '__main__':
    class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass
    
    
    # Parse the arguments from cli
    parser = argparse.ArgumentParser(description="Trains the source separation model.",
                                     formatter_class=MyFormatter)
    parser.add_argument("run_name",
                        help="Name to give to this run in visdom and to save the model as.")
    parser.add_argument("dataset_root",
                        help="Path to a directory containing the 'pop_midi_dataset_ismir' dataset "
                             "and the index files")
    parser.add_argument("source_instruments",
                        help="Comma-separated list of the names of instruments for the input"
                             "samples. For a complete list of available instruments: python -m "
                             "source_separation.data_objects.midi_instruments")
    parser.add_argument("target_instruments",
                        help="Identical to source_instruments but for the target instruments to"
                             "predict. All target instruments must appear as source intruments.")
    parser.add_argument("-s", "--save_every", default=100, type=int,
                        help="Number of steps between updates of the model on the disk. Set to 0"
                             "to disable saving the model.")
    parser.add_argument("-r", "--chunk_reuse", default=2, type=int,
                        help="Number of times a chunk is reused in training. Higher: more data-"
                             "efficient but also more redundancy. Increase this value if ")
    parser.add_argument("-p", "--chunk_pool_size", default=1000, type=int,
                        help="Size of the chunk pool. Higher means more domain variance in the "
                             "input batches, but higher RAM usage (and disk if quickstart is set)."
                             "Warning: the pool size is not a strict upper bound. The pool can be "
                             "filled to more than 100% of its size. With all default parameters, "
                             "a full pool takes 1.7gb of space.")
    parser.add_argument("-q", "--quickstart", action="store_true",
                        help="If set, the first chunk pool will be cached to disk and reused at"
                             "the beggining of subsequent trainings. If not, the pool must be "
                             "refilled before each training.")
    parser.add_argument("-d", "--chunk_duration", default=5, type=int,
                        help="Duration of the chunks, in seconds")
    parser.add_argument("-b", "--batch_size", default=32, type=int)
    parser.add_argument("-l", "--learning_rate_init", default=0.01, type=float)
    
    # Format the arguments
    args = parser.parse_args()
    args.dataset_root = Path(args.dataset_root)
    get_instruments_id = lambda l: list(map(get_instrument_id, l.split(",")))
    args.source_instruments = get_instruments_id(args.source_instruments)
    args.target_instruments = get_instruments_id(args.target_instruments)

    # Update the hparams with CLI arguments
    hparams.update(**vars(args))
    
    # Begin the training
    train(args, hparams)
