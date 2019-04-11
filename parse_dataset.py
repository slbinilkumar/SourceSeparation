from data_objects import Music
from glob import glob
from tqdm import tqdm
import os

# TODO: put in a config file
dataset_root = r"E:\Datasets\Midi"

index_fpath = os.path.join(dataset_root, "midi_index.txt")
index_file = open(index_fpath, 'a', encoding='utf-8')
mid_fpaths = glob(os.path.join(dataset_root, "**/*.mid"), recursive=True)
valid_count = 0
for fpath in tqdm(mid_fpaths, "Parsing midi files"):
    try:
        music = Music(fpath)
    except:
        continue
    rel_fpath = os.path.relpath(fpath, dataset_root)
    index_file.write(f"\"{rel_fpath}\":{','.join(map(str, music.all_instruments))}\n")
    valid_count += 1

print("Parsed %d files, %d valid MIDI files found (%.1f%%)" % 
      (len(mid_fpaths), valid_count, 100 * valid_count / len(mid_fpaths)))
index_file.close()




