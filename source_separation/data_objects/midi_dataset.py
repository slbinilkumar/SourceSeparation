from source_separation.data_objects.music import Music
from pathos.threading import ThreadPool
from sklearn.utils import shuffle
from pathlib import Path
from typing import List
from time import perf_counter as timer
import numpy as np


class MidiDataset:
    def __init__(self, root: Path, is_train: bool, hparams):
        """
        Creates a dataset that synthesizes instrument tracks from midi files. Call 
        MidiDataset.generate() to iterate over the dataset and retrieve pairs of fixed-size 
        segments (called chunks) from the generated waveforms, with only the instruments
        selected playing.
        
        :param root: path to the directory containing midi files and the corresponding index 
        files generated from parse_dataset.py.
        :param is_train: if True, the train index file is used. Otherwise, the test index file 
        will be used. 
        """
        self.hparams = hparams
        self.epochs = 0
        self.epoch_progress = 0.
        self.musics_sampled = 0
        self.chunks_generated = 0

        
        # Build the index: a list of tuples (fpath, instruments) 
        index_fname = "midi_%s_index.txt" % ("train" if is_train else "test")
        index_fpath = root.joinpath(index_fname)
        with index_fpath.open("r") as index_file:
            index = [line.split(":") for line in index_file]
        self.index = [(root.joinpath(fpath.replace('"', '')),
                      list(map(int, instruments.split(',')))) for fpath, instruments in index]

    def generate(self, source_instruments: List[int], target_instruments: List[int],
                 batch_size: int, n_threads=4, chunk_reuse_factor=1, chunk_pool_size=1000, 
                 quickstart=False):
        # Todo: redo the doc
        # """
        # :param chunk_duration: the duration, in seconds, of the audio segments that the dataset 
        # yields.
        # :param source_instruments: a list of instrument IDs. Only musics containing all these 
        # instruments will be sampled from, and only the tracks for these instruments will be 
        # generated for the first output of the dataset.
        # :param target_instruments: a list of instrument IDs. All instruments must also be 
        # contained in <source_instruments>. Only the tracks for these instruments will be 
        # generated for the second output of the dataset.
        # :param batch_size: batch size for the data generated by the dataset.
        # :param n_threads: number of threads used for synthesizing the midi files.
        # """
        # Todo: later, think about how the instruments should be selected
        assert all((i in source_instruments) for i in target_instruments), \
            "Some target instruments are not in the set of the source instruments."
        assert chunk_pool_size >= batch_size, \
            "The chunk pool size should be greater or equal to the batch size."
        
        # Reset all generation statistics
        self.epochs = 0
        self.epoch_progress = 0.
        self.musics_sampled = 0
        self.chunks_generated = 0
        
        # Create a generator that loops infinitely over the songs in a random order
        def midi_fpath_generator():
            midi_fpaths = list(self._get_files_by_instruments(source_instruments))
            while True:
                shuffle(midi_fpaths)
                for i, midi_fpath in enumerate(midi_fpaths, 1):
                    yield midi_fpath
                    self.epoch_progress = i / len(midi_fpaths) 
                self.epochs += 1
        midi_fpath_generator = midi_fpath_generator()
        
        # Define a function to fill a buffer
        def begin_next_buffer():
            # Estimate how many musics to sample from to generate a full batch
            avg_n_chunks = self.chunks_generated / self.musics_sampled if self.musics_sampled else 0
            n_musics = int(np.ceil(batch_size / avg_n_chunks) if avg_n_chunks else 0) + n_threads
            self.musics_sampled += n_musics
            
            # Begin filling the buffer with threads from the threadpool 
            func = lambda fpath: self._extract_chunks(fpath, source_instruments, target_instruments)
            midi_fpaths = [next(midi_fpath_generator) for _ in range(n_musics)]
            return thread_pool.uimap(func, midi_fpaths)
        
        # Define a function the fill the chunk pool
        def refill_chunk_pool(chunk_pool, chunk_pool_uses, buffer):
            # Do nothing if the pool is already full
            if len(chunk_pool) >= chunk_pool_size:
                return chunk_pool, chunk_pool_uses, buffer
            
            while len(chunk_pool) < chunk_pool_size:
                # Retrieve the elements from the next buffer that were generated in the
                # background. If it is not done generating, block until so with a call to list().
                print("Refilling the chunk pool... ", end=" ")
                start = timer()
                buffer = list(buffer)
                
                # Flatten the buffer to retrieve a list of chunks, and append all the contents of 
                # the buffer to the chunk pool
                n_musics = len(buffer)
                buffer = [chunk for chunks in buffer for chunk in chunks if chunk.shape != (0,)]
                chunk_pool.extend(buffer)
                chunk_pool_uses.extend([chunk_reuse_factor] * len(buffer))
                delta = timer() - start   
                print("Done!\nBlocked %dms to generate %d chunks from %d musics. The pool is now "
                      "%.0f%% full." % (int(delta * 1000), len(buffer), n_musics,
                                        100 * len(chunk_pool) / chunk_pool_size))
                
                # Register statistics about the number of generated chunks to better estimate how
                # many jobs will be needed to fill the pool the next time
                self.chunks_generated += len(buffer)
    
                # Begin a new buffer in the background
                buffer = begin_next_buffer()
                
            # Shuffle the chunk pool so as to mix different musics in a same batch
            chunk_pool, chunk_pool_uses = shuffle(chunk_pool, chunk_pool_uses)
            return chunk_pool, chunk_pool_uses, buffer
        
        # Create the threadpool, the chunk pool and initialize the buffers
        thread_pool = ThreadPool(n_threads)
        chunk_pool = []
        chunk_pool_uses = []
        buffer = begin_next_buffer()
        
        # If quickstart is enabled, load a precomputed pool from disk or build one if it 
        # doesn't exist yet.
        if quickstart:
            quickstart_id = "_%s_%s" % ("-".join(map(str, source_instruments)),
                                        "-".join(map(str, target_instruments)))
            chunk_pool_fpath = Path("quickstart%s.npy" % quickstart_id)
            if not chunk_pool_fpath.exists():
                chunk_pool, chunk_pool_uses, buffer = \
                    refill_chunk_pool(chunk_pool, chunk_pool_uses, buffer)
                np.save(chunk_pool_fpath, chunk_pool[:chunk_pool_size])
                print("Saved the quickstart pool to the disk!")
            else:
                print("Loading from the quickstart chunk pool.")
                chunk_pool = list(np.load(chunk_pool_fpath))
                chunk_pool_uses = [chunk_reuse_factor] * chunk_pool_size
        
        # We wrap the generator inside an explicit generator function. We could simply make this 
        # function (MidiDataset.generate()) the generator itself, but splitting the initialization
        # code and the actual generator allows us to execute the initialization when 
        # MidiDataset.generate() is called for the first time, rather than when we start iterating
        # from the dataset.
        def generator(chunk_pool, chunk_pool_uses, buffer):
            while True:
                # Make sure the chunk pool is full
                chunk_pool, chunk_pool_uses, buffer = \
                    refill_chunk_pool(chunk_pool, chunk_pool_uses, buffer)
            
                # Consume elements from the chunk pool to generate a batch
                batch = chunk_pool[:batch_size]
                batch_uses = chunk_pool_uses[:batch_size]
                del chunk_pool[:batch_size]
                del chunk_pool_uses[:batch_size]
                for chunk, chunk_uses in zip(batch, batch_uses):
                    if chunk_uses - 1 == 0:
                        continue
                    chunk_pool.append(chunk)
                    chunk_pool_uses.append(chunk_uses - 1)

                # Yield the chunks as a batch
                yield np.array(batch).transpose((1, 0, 2))
                
        return generator(chunk_pool, chunk_pool_uses, buffer)

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
            if selector((i in midi_instruments) for i in instruments):
                yield midi_fpath
                
    def _debug_compare_chunks(self, source_chunk, target_chunk):
        zero_prop = lambda chunk: np.sum(np.abs(chunk) < self.hparams.silence_threshold) \
                                  / len(chunk)
        zero_source_prop = zero_prop(source_chunk)
        zero_target_prop = zero_prop(target_chunk)
        similarity_prop = zero_prop(source_chunk - target_chunk)
        np.sum(np.abs(source_chunk - target_chunk) < self.hparams.silence_threshold) /\
            len(target_chunk)
        print("Proportion in the source that is silence: %.2f%%" % (zero_source_prop * 100))
        print("Proportion in the target that is silence: %.2f%%" % (zero_target_prop * 100))
        print("Proportion of the chunk that is equal: %.2f%%" % (similarity_prop * 100))
    
    def _extract_chunks(self, midi_fpath, source_instruments, target_instruments):
        # Load the midi file from disk
        music = Music(sample_rate=self.hparams.sample_rate, fpath=str(midi_fpath))
        
        # Ignore songs that are too long (prevents corrupted midis from blocking the sampling)
        if music.mid.length > self.hparams.max_midi_duration:
            return np.array([]), np.array([]) 

        # Generate a waveform for the reference (source) audio and the target audio the network
        # had to produce.
        source_wav = music.generate_waveform(source_instruments)
        target_wav = music.generate_waveform(target_instruments)
        assert len(source_wav) == len(target_wav)
        
        # Pad waveforms to a multiple of the chunk size
        chunk_size = self.hparams.chunk_duration * self.hparams.sample_rate
        padding = chunk_size - (len(source_wav) % chunk_size)
        source_wav = np.pad(source_wav, (0, padding), "constant")
        target_wav = np.pad(target_wav, (0, padding), "constant")
        
        # Iterate over the waveforms to create chunks
        chunks = []
        for i in range(0, len(source_wav), chunk_size):
            source_chunk, target_chunk = source_wav[i:i + chunk_size], target_wav[i:i + chunk_size]
            
            # Compute what proportion of the waveform is repeated without change in the target 
            # waveform. This only applies if we're not predicting the identity.
            if not np.array_equal(source_instruments, target_instruments):
                abs_diff = np.abs(source_chunk - target_chunk)
                equal_prop = np.sum(abs_diff < self.hparams.silence_threshold) / len(abs_diff)
                if equal_prop >= self.hparams.chunk_equal_prop_max:
                    continue
                
            # Compute what proportion of the target waveform is silence.
            silence_prop = np.sum(np.abs(target_chunk) < self.hparams.silence_threshold) \
                           / len(target_chunk)
            if silence_prop >= self.hparams.chunk_silence_prop_max:
                continue
                
            # Accept samples that were not discarded
            chunks.append(np.array([source_chunk, target_chunk]))
            
        return chunks
