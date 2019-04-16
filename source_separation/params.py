from pathlib import Path

# TODO: rewrite with hparams from tensorflow
dataset_root = Path(r"E:\Datasets\Midi")
sample_rate = 44100
package_root = Path(__file__).parent.absolute()

# Mutually exclusive --n_instruments or --instrument_list
