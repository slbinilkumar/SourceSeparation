from source_separation.hparams import hparams
from source_separation.train import train
from pathlib import Path
import argparse


if __name__ == '__main__':
    class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass
    
    
    # Parse the arguments from cli
    default_instruments = ",".join(map(str, hparams.default_instruments))
    parser = argparse.ArgumentParser(description="Trains the source separation model.",
                                     formatter_class=MyFormatter)
    parser.add_argument("run_name",
                        help="Name to give to this run in visdom and to save the model as.")
    parser.add_argument("dataset_root",
                        help="Path to a directory containing the 'pop_midi_dataset_ismir' dataset "
                             "and the index files")
    parser.add_argument("-i", "--instruments", default=default_instruments, type=str,
                        help="Comma-separated list of instruments ids. For a complete list of "
                             "available instruments: python -m "
                             "source_separation.data_objects.midi_instruments")
    parser.add_argument("-s", "--save_every", default=100, type=int,
                        help="Number of steps between updates of the model on the disk. Set to 0"
                             "to disable saving the model.")
    parser.add_argument("-v", "--vis_every", default=100, type=int,
                        help="Number of steps between generating visualizations of the generated "
                             "waveforms.")
    parser.add_argument("-r", "--chunk_reuse", default=1, type=int,
                        help="Number of times a chunk is reused in training. Higher: more data-"
                             "efficient but also more redundancy. Increase this value if ")
    parser.add_argument("-p", "--pool_size", default=8, type=int,
                        help="Size of the chunk pool. Higher means more domain variance in the "
                             "input batches, but higher RAM usage. Set this to the batch size if"
                             "chunk_reuse is set to 1, otherwise grow this parameter.")
    parser.add_argument("-q", "--quickstart", action="store_true",
                        help="If set, the first chunk pool will be cached to disk and reused at"
                             "the beggining of subsequent trainings. If not, the pool must be "
                             "refilled before each training.")
    parser.add_argument("-d", "--chunk_duration", default=0.1, type=float,
                        help="Duration of the chunks, in seconds")
    parser.add_argument("-b", "--batch_size", default=8, type=int)
    parser.add_argument("-l", "--learning_rate_init", default=0.01, type=float)
    
    # Format the arguments
    args = parser.parse_args()
    args.dataset_root = Path(args.dataset_root)
    args.instruments = list(map(int, args.instruments.split(",")))

    # Update the hparams with CLI arguments
    hparams.update(**vars(args))
    
    # Begin the training
    train(args, hparams)
