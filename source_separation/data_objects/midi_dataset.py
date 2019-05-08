from source_separation.data_objects.music import Music
from source_separation.hparams import HParams
from pathos.threading import ThreadPool
from sklearn.utils import shuffle
from pathlib import Path
from typing import List
from time import perf_counter as timer
import numpy as np
import torch


class MidiDataset:
    def __init__(self, root: Path, is_train: bool, chunk_size: int, hparams: HParams):
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
        self.chunk_size = chunk_size
        self.hparams = hparams
        self.epochs = 0
        self.epoch_progress = 0.
        self.musics_sampled = 0
        self.chunks_generated = 0
        
        self.debug_midi_fpaths = []

        # Build the index: a list of tuples (fpath, instruments) 
        index_fname = "midi_%s_index.txt" % ("train" if is_train else "test")
        index_fpath = root.joinpath(index_fname)
        with index_fpath.open("r") as index_file:
            index = [line.split(":") for line in index_file]
        self.index = [(root.joinpath(fpath.replace('"', '')),
                      list(map(int, instruments.split(',')))) for fpath, instruments in index]

    def generate(self, instruments: List[int], batch_size: int, n_threads=4, 
                 max_chunks_per_music=-1, chunk_reuse=1, chunk_pool_size=1000):
        """
        Creates a generator that iterates over the dataset to generate chunks. The generator 
        first starts will filling a pool of chunks. 
        
        :param instruments: the id of the instruments to keep when generating chunks
        :param batch_size: the size of the batches yielded
        :param n_threads: the number of threads to synthesize waveforms in parallel
        :param chunk_reuse: the number of times a single chunk will be used per epoch
        :param chunk_pool_size: the minimum number of chunks the pool must contain before 
        starting to yield batches
        :return: 
        """
        assert chunk_pool_size >= batch_size, \
            "The chunk pool size should be greater or equal to the batch size."
        
        # Reset all generation statistics
        self.epochs = 0
        self.epoch_progress = 0.
        self.musics_sampled = 0
        self.chunks_generated = 0
        
        # Create a generator that loops infinitely over the songs in a random order
        def midi_fpath_generator():
            midi_fpaths = list(self._get_files_by_instruments(instruments, at_least=2))
            midi_fpaths = shuffle(midi_fpaths)
            while True:
                for i, midi_fpath in enumerate(midi_fpaths, 1):
                    yield midi_fpath
                    self.debug_midi_fpaths.append(midi_fpath)
                    if len(self.debug_midi_fpaths) > n_threads * 2:
                        del self.debug_midi_fpaths[0]
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
            func = lambda fpath: self._extract_chunks(fpath, instruments, max_chunks_per_music)
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
                start = timer()
                buffer = list(buffer)
                
                # Flatten the buffer to retrieve a list of chunks, and append all the contents of 
                # the buffer to the chunk pool
                n_musics = len(buffer)
                buffer = [chunk for chunks in buffer for chunk in chunks]
                chunk_pool.extend(buffer)
                chunk_pool_uses.extend([chunk_reuse] * len(buffer))
                delta = timer() - start   
                print("Blocked %dms to generate %d chunks from %d musics." % 
                      (int(delta * 1000), len(buffer), n_musics))
                
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
                chunks = chunk_pool[:batch_size]
                chunks_uses = chunk_pool_uses[:batch_size]
                del chunk_pool[:batch_size]
                del chunk_pool_uses[:batch_size]
                for chunk, chunk_uses in zip(chunks, chunks_uses):
                    if chunk_uses == 1:
                        continue
                    chunk_pool.append(chunk)
                    chunk_pool_uses.append(chunk_uses - 1)

                # Yield the chunks as a batch
                yield self._collate(chunks, instruments)
                
        return generator(chunk_pool, chunk_pool_uses, buffer)
    
    def _collate(self, chunks, instruments):
        """
        Collates chunks into a batch.
        """
        # Expand the target to also contain instruments that do not appear in the music
        x = np.array([chunk[0] for chunk in chunks])
        y = np.zeros((x.shape[0], len(instruments), x.shape[1]), dtype=np.float32)
        for i in range(len(chunks)):
            for instrument_chunk, instrument in zip(chunks[i][1], chunks[i][2]):
                index = instruments.index(instrument)
                y[i, index] = instrument_chunk
        
        return torch.from_numpy(x), torch.from_numpy(y)

    def _get_files_by_instruments(self, instruments, at_least=2):
        """
        Yields midi filepaths in the dataset that contain a specific set of instruments.

        :param instruments: a list of instrument IDs.
        :param at_least: how many instruments must at least match between the list of instruments
        and the midi's instruments for the midi to be selected. 
        :return: a generator that yields the midi filepaths as strings
        """
        for midi_fpath, midi_instruments in self.index:
            if sum((i in midi_instruments) for i in instruments) >= at_least:
                yield midi_fpath
    
    def _extract_chunks(self, midi_fpath, instruments, max_chunks_per_music=-1):
        # Load the midi file from disk
        try:
            music = Music(sample_rate=self.hparams.sample_rate, fpath=str(midi_fpath))
        except:
            return []
        
        # Ignore songs that are too long (prevents corrupted midis from blocking the sampling)
        if music.mid.length > self.hparams.max_midi_duration:
            return []

        # Generate a waveform for the reference (source) audio and the target audio the network
        # had to produce.
        instruments = [i for i in music.all_instruments if i in instruments]
        try:
            target_wavs = [music.generate_waveform([i]) for i in instruments]
        except:
            return []
        
        # Pad waveforms to a multiple of the chunk size
        padding = self.chunk_size - (len(target_wavs[0]) % self.chunk_size)
        target_wavs = np.array([np.pad(wav, (0, padding), "constant") for wav in target_wavs])
        
        # Iterate over the waveforms to create chunks
        chunks = []
        for i in shuffle(range(0, len(target_wavs[0]), self.chunk_size)):
            # Cut a chunk from the waveform, and sum these chunks to create the mixed source signal
            target_chunks = np.array([wav[i:i + self.chunk_size] for wav in target_wavs])
            source_chunk = np.sum(target_chunks, axis=0)
            
            # Normalize chunks by the same constant, so that they remain within [-1, 1] and that
            # the sum of the target chunks remains exactly equal to the source chunk.
            max_sample = np.abs(source_chunk).max()
            if max_sample <= self.hparams.silence_threshold:
                continue
            target_chunks /= max_sample
            source_chunk /= max_sample
            
            # Discard chunks that don't have enough playing time in them. See hparams for a
            # description of this process.
            sum_prop_play = np.sum((np.abs(target_chunks) >= self.hparams.silence_threshold) /
                            target_chunks.shape[1])
            if sum_prop_play <= self.hparams.chunk_sum_prop_play_min:
                continue
            
            # Accept samples that were not discarded
            chunks.append((source_chunk, target_chunks, instruments))
            
            if max_chunks_per_music != -1 and len(chunks) >= max_chunks_per_music:
                break
            
        return chunks
