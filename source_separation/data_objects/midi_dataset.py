from tensorflow.python.data.ops.dataset_ops import DatasetV1Adapter 
from source_separation.data_objects.music import Music
from source_separation.params import sample_rate
from pathlib import Path
from typing import List
import tensorflow as tf
import numpy as np


# Samples below this value are considered to be 0
silence_threshold = 1e-4        

# If more than this proportion of the waveform is equivalent between the source and the target 
# waveform, the sample is discarded.
chunk_equal_prop_max = 0.65 
# If more than this proportion of the target sample is silence, the sample is discarded.
chunk_silence_prop_max = 0.4


class MidiDataset(DatasetV1Adapter):
    def __init__(self, root: Path, is_train: bool, chunk_duration: int, 
                 source_instruments: List[int], target_instruments: List[int], batch_size: int,
                 n_threads=4, shuffle_buffer=512):
        """
        Creates a dataset subclassing tf.data.Dataset that synthesizes instrument tracks from midi
        files. The instruments to generate must be selected. Yields pairs of fixed-size segments 
        (called chunks) from these waveforms.
        
        :param root: path to the directory containing midi files and the corresponding index 
        files generated from parse_dataset.py.
        :param is_train: if True, the train index file is used. Otherwise, the test index file 
        will be used. 
        :param chunk_duration: the duration, in seconds, of the audio segments that the dataset 
        yields.
        :param source_instruments: a list of instrument IDs. Only musics containing all these 
        instruments will be sampled from, and only the tracks for these instruments will be 
        generated for the first output of the dataset.
        :param target_instruments: a list of instrument IDs. All instruments must also be 
        contained in <source_instruments>. Only the tracks for these instruments will be 
        generated for the second output of the dataset.
        :param batch_size: batch size for the data generated by the dataset.
        :param n_threads: number of threads used for synthesizing the midi files.
        Todo: replace the shuffle buffer argument by an argument that dictates how many different
            musics are in the buffer at once. Then, approximate the buffer size from this argument,
            from the sample duration and from the average length of songs (keep it an 
            approximation: the buffer size remains constant at execution).
        :param shuffle_buffer: size of the buffer used for randomization 
        """
        self.root = root
        self.chunk_duration = chunk_duration
        self.log_chunks_accepted = 0
        self.log_chunks_total = 0
        
        # Todo: later, think about how the instruments should be selected
        self.source_instruments = source_instruments
        self.target_instruments = target_instruments
        assert all((i in source_instruments) for i in target_instruments), \
            "Some target instruments are not in the set of the source instruments."
        
        # Build the index: a list of tuples (fpath, instruments) 
        self.index = self._build_index(root, is_train)
        
        # Create the tf.data.Dataset object
        dataset = self._create_dataset(batch_size, n_threads, shuffle_buffer)
        super().__init__(dataset)
        
    def _build_index(self, root: Path, is_train: bool):
        """
        Yields midi filepaths in the dataset that contain a specific set of instruments.
        """
        index_fname = "midi_%s_index.txt" % ("train" if is_train else "test")
        index_fpath = root.joinpath(index_fname)
        with index_fpath.open("r") as index_file:
            index = [line.split(":") for line in index_file]
        index = [(root.joinpath(fpath.replace('"', '')), 
                  list(map(int, instruments.split(',')))) for fpath, instruments in index]
        return index
        
    def _get_files_by_instruments(self, instruments, mode="and"):
        """
        Yields midi filepaths in the dataset that contain a specific set of instruments.

        :param instruments: a list of instrument IDs.
        :param mode: if "and", the midi file will only be returned if all the instruments 
        required are found in the midi. If "or", it will be returned as soon as one instrument is 
        found in the midi.  
        :return: a generator that yields the midi filepaths as strings
        """
        for midi_fpath, midi_instruments in self.index:
            selector = all if mode == "and" else any
            # If this line causes PyCharm to throw SystemErrors, add PYDEVD_USE_FRAME_EVAL with 
            # value NO as environment variable in the config 
            if selector((i in midi_instruments) for i in instruments):
                yield midi_fpath
                
    @staticmethod
    def _debug_compare_chunks(source_chunk, target_chunk):
        zero_prop = lambda chunk: np.sum(np.abs(chunk) < silence_threshold) / len(chunk)
        zero_source_prop = zero_prop(source_chunk)
        zero_target_prop = zero_prop(target_chunk)
        similarity_prop = zero_prop(source_chunk - target_chunk)
        np.sum(np.abs(source_chunk - target_chunk) < silence_threshold) / len(target_chunk)
        print("Proportion in the source that is silence: %.2f%%" % (zero_source_prop * 100))
        print("Proportion in the target that is silence: %.2f%%" % (zero_target_prop * 100))
        print("Proportion of the chunk that is equal: %.2f%%" % (similarity_prop * 100))
        
    def _debug_accept_rate(self):
        print("Accepted: %6d   Total: %6d" % (self.log_chunks_accepted, self.log_chunks_total))
    
    def _create_sample(self, midi_fpath):
        # Load the midi file from disk
        if isinstance(midi_fpath, tf.Tensor):
            midi_fpath = midi_fpath.numpy().decode()
        music = Music(midi_fpath)
        
        # Generate a waveform for the reference (source) audio and the target audio the network
        # had to produce.
        source_wav = music.generate_waveform(self.source_instruments)
        target_wav = music.generate_waveform(self.target_instruments)
        assert len(source_wav) == len(target_wav)
        
        # Pad waveforms to a multiple of the chunk size
        chunk_size = self.chunk_duration * sample_rate
        padding = chunk_size - (len(source_wav) % chunk_size)
        source_wav = np.append(source_wav, np.zeros(padding))
        target_wav = np.append(target_wav, np.zeros(padding))
        
        # Iterate over the waveforms to create chunks
        source_chunks, target_chunks = [], []
        for i in range(0, len(source_wav), chunk_size):
            source_chunk, target_chunk = source_wav[i:i + chunk_size], target_wav[i:i + chunk_size]
            self.log_chunks_total += 1
            
            # Compute what proportion of the waveform is repeated without change in the target 
            # waveform. 
            abs_diff = np.abs(source_chunk - target_chunk)
            equal_prop = np.sum(abs_diff < silence_threshold) / len(abs_diff)
            if equal_prop >= chunk_equal_prop_max:
                continue
                
            # Compute what proportion of the target waveform is silence.
            silence_prop = np.sum(np.abs(target_chunk) < silence_threshold) / len(target_chunk)
            if silence_prop >= chunk_silence_prop_max:
                continue
                
            # Accept samples that were not discarded
            self.log_chunks_accepted += 1
            source_chunks.append(source_chunk)
            target_chunks.append(target_chunk)
            
        return np.array(source_chunks), np.array(target_chunks)
        
    def _create_dataset(self, batch_size: int, n_threads: int, shuffle_buffer: int):
        # The source is a generator of midi filepaths (converted to tensorflow string tensors)
        def generator():
            for fpath in self._get_files_by_instruments(self.source_instruments):
                yield tf.constant(str(fpath), dtype=tf.string)
        dataset = tf.data.Dataset.from_generator(generator, tf.string)
        
        # The midis are then loaded in memory, a source and a target waveform is generated for each,
        # and these waveforms are split in chunks of equal size.
        dataset = dataset.map(lambda midi_fpath: tf.py_function(
            self._create_sample,
            (midi_fpath,),
            (tf.float32, tf.float32)
        ), n_threads)
        # Todo: if needed for debugging purposes, attach the source midi filepath to each chunk.
        
        # We flatten the dataset to obtain a dataset of chunks. It is important to do this step
        # before shuffling, so as to allow chunks from different songs to be mixed in a same batch.
        dataset = dataset.flat_map(lambda x, y: (tf.data.Dataset.from_tensor_slices((x, y))))
    
        # Todo: analyze whether it's better to have repeat here, or just after the line 
        #   Dataset.from_generator. I think it does not make any difference, because all the
        #   functions chained above are generators, but I'm not 100% sure.
        #   Note that repeat must come before batch and before shuffle.
        #   Also, shuffle should be before batch.
        # A bigger buffer gives a better randomization but takes more memory
        dataset = dataset.repeat().shuffle(buffer_size=shuffle_buffer)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) # Is it going to sing?
    
        return dataset
