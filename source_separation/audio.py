from typing import List
import numpy as np


def get_active_mask(wavs):
    """
    Computes a binary mask that corresponds to whether at least one of the waveforms is outputting
    audio.
    
    TODO: Include a minimum duration such that no active area is smaller than this duration,  
        areas that are too small are either removed or included within a larger one  
        

    :param wavs: an array-like object containing multiple waveforms as numpy arrays of shape
    (n_samples,) each.
    :return: the mask as a numpy array of booleans of shape (n_samples,). True indicates that 
    sound is being played for this sample. 
    """
    
    mask = np.sum(wav.astype(np.bool) for wav in wavs)
    return mask
    
