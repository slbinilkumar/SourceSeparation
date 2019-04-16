from source_separation.data_objects.music import Music
from source_separation.params import sample_rate
from pathlib import Path
from typing import List
import tensorflow as tf
import numpy as np


class MidiDataset:
    def __init__(self, root: Path, is_train: bool, sample_duration, 
                 source_instruments: List[int],
                 target_instruments: List[int]):
        self.root = root
        self.sample_duration = sample_duration
        
        # Todo: later, think about how the instruments should be selected
        self.source_instruments = source_instruments
        self.target_instruments = target_instruments
        assert all((i in source_instruments) for i in target_instruments), \
            "Some target instruments are not in the set of the source instruments."
        
        # Build the index: a list of tuples (fpath, instruments) 
        index_fname = "midi_%s_index.txt" % ("train" if is_train else "test")
        index_fpath = root.joinpath(index_fname)
        with index_fpath.open("r") as index_file:
            self.index = [line.split(":") for line in index_file]
        self.index = [(root.joinpath(fpath.replace('"', '')), 
                       list(map(int, instruments.split(',')))) for fpath, instruments in self.index]
        
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
            if selector((i in instruments) for i in midi_instruments):
                yield midi_fpath
    
    def waveform_to_chunks(self, wav):
        chunk_length = self.sample_duration * sample_rate
        padding = chunk_length - (len(wav) % chunk_length)
        wav = np.append(wav, np.zeros(padding))
        return np.vstack(np.split(wav, len(wav) / chunk_length))
    
    def create_sample(self, midi_fpath):
        # Load the midi file from disk
        if isinstance(midi_fpath, tf.Tensor):
            midi_fpath = midi_fpath.numpy().decode()
        music = Music(midi_fpath)
        
        # Generate a waveform for the reference (source) audio and the target audio the network
        # had to produce.
        # TODO: use the audio mask (see audio.py) so as to not generate too much audio with
        #   no instruments played.
        source_wav = music.generate_waveform(self.source_instruments)
        target_wav = music.generate_waveform(self.target_instruments)
        assert len(source_wav) == len(target_wav)
        
        # Split the waveforms into short chunks of equal duration
        source_chunks = self.waveform_to_chunks(source_wav) 
        target_chunks = self.waveform_to_chunks(target_wav)
        assert source_wav.shape == target_wav.shape

        return source_chunks, target_chunks
        
    def generate_dataset(self, batch_size, n_threads=4, shuffle_buffer=512):
        """
        Generates a tf.data.Dataset using midi files located in the index file
        
        :param batch_size: size of the batch
        :param sample_duration: duration of one sample in seconds
        Todo: replace the shuffle buffer argument by an argument that dictates how many different
            musics are in the buffer at once. Then, approximate the buffer size from this argument,
            from the sample duration and from the average length of songs (keep it an 
            approximation: the buffer size remains constant at execution).
        :param shuffle_buffer: size of the buffer used for randomization
        :return: an instance of a tf.data.Dataset
        """
        # The source is a generator of midi filepaths (converted to tensorflow string tensors)
        def generator():
            for fpath in self._get_files_by_instruments(self.source_instruments):
                yield tf.constant(str(fpath), dtype=tf.string)
        dataset = tf.data.Dataset.from_generator(generator, tf.string)
        
        # The midis are then loaded in memory, a source and target waveform is generated for each,
        # and these waveforms are split in chunks of equal size.
        dataset = dataset.map(lambda midi_fpath: tf.py_function(
            self.create_sample,
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
