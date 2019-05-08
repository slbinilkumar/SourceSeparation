from source_separation.data_objects import Music, get_instrument_name
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Displays the distribution of instruments in the "
                                                 "dataset.")
    parser.add_argument("dataset_root",
                        help="Path to the directory containing your generated index files.")
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    
    # Gather the distribution of instruments
    instrument_counts = np.zeros(129)
    instrument_ids = np.concatenate((np.arange(128), [-1]))
    n_instruments = 0
    for name in ["train", "test"]:
        file = dataset_root.joinpath("midi_%s_index.txt" % name).open('r', encoding='utf-8')
        for line in file:
            instruments = list(map(int, line.split(":")[1].rstrip().split(",")))
            instrument_counts[instruments] += 1
            n_instruments += 1
            
    # Sort the instruments
    indices = np.argsort(instrument_counts)[::-1]
    instrument_counts = instrument_counts[indices]
    instrument_ids = instrument_ids[indices]
    instrument_names = list(map(get_instrument_name, instrument_ids))
    
    # Display statistics
    print("Total number of musics in the dataset: %d" % n_instruments)
    plt.bar(instrument_names[:10], instrument_counts[:10])
    plt.title("Ten most present instruments in the dataset")
    plt.ylabel("Count")
    plt.show()
    
    for instrument_id, instrument_name in zip(instrument_ids, instrument_names[:20]):
        print("%s: %d" % (instrument_name, instrument_id))
    
    
    


