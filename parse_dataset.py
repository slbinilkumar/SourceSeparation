from source_separation.data_objects import Music
from random import random
from glob import glob
from tqdm import tqdm
import os

# TODO: put in a config file
dataset_root = r"E:\Datasets\Midi"

# Create a test and train index
test_prop = 0.2
train_index_fpath = os.path.join(dataset_root, "midi_train_index.txt")
train_index_file = open(train_index_fpath, 'w', encoding='utf-8')
test_index_fpath = os.path.join(dataset_root, "midi_test_index.txt")
test_index_file = open(test_index_fpath, 'w', encoding='utf-8')

# Parse all files in the dataset
mid_fpaths = glob(os.path.join(dataset_root, "**/*.mid"), recursive=True)
valid_count = 0
for fpath in tqdm(mid_fpaths, "Parsing midi files"):
    # Exclude files that fail to be parsed
    try:
        music = Music(fpath)
        # There are a very few files that have no instruments at all (and thus no sound)
        assert len(music.all_instruments) > 0
    except:
        continue
        
    # Index the file to either the train or test set
    rel_fpath = os.path.relpath(fpath, dataset_root)
    line = f"\"{rel_fpath}\":{','.join(map(str, music.all_instruments))}\n"
    (test_index_file if random() < test_prop else train_index_file).write(line)
    valid_count += 1

print("Parsed %d files, %d valid MIDI files found (%.1f%%)" % 
      (len(mid_fpaths), valid_count, 100 * valid_count / len(mid_fpaths)))
test_index_file.close()
train_index_file.close()



