from source_separation.data_objects import Music
from source_separation.hparams import hparams
from pathlib import Path
from random import random
from tqdm import tqdm
import argparse

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Indexes the files in a MIDI dataset.")
    parser.add_argument("dataset_root",
                        help="Path to the root directory of any MIDI corpus. The MIDI files can be "
                             "present alongside other types of files, and can be nested in "
                             "directories.")
    parser.add_argument("-p", "--test_prop", default=0.2, type=float)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    test_prop = args.test_prop
    
    # Create a test and train index
    train_index_fpath = dataset_root.joinpath("midi_train_index.txt")
    train_index_file = open(train_index_fpath, 'w', encoding='utf-8')
    test_index_fpath = dataset_root.joinpath("midi_test_index.txt")
    test_index_file = open(test_index_fpath, 'w', encoding='utf-8')
    
    # Parse all files in the dataset
    mid_fpaths = dataset_root.rglob("**/*.mid")
    valid_count = 0
    for fpath in tqdm(mid_fpaths, "Parsing midi files"):
        # Exclude files that fail to be parsed
        try:
            music = Music(hparams.sample_rate, fpath=fpath)
            # There are a very few files that have no instruments at all (and thus no sound)
            assert len(music.all_instruments) > 0
        except Exception as e:
            print(fpath)
            continue
            
        # Index the file to either the train or test set
        rel_fpath = fpath.relative_to(dataset_root)
        line = f"\"{rel_fpath}\":{','.join(map(str, music.all_instruments))}\n"
        (test_index_file if random() < test_prop else train_index_file).write(line)
        valid_count += 1
    
    print("Parsed %d files, %d valid MIDI files found (%.1f%%)" % 
          (len(mid_fpaths), valid_count, 100 * valid_count / len(mid_fpaths)))
    test_index_file.close()
    train_index_file.close()



