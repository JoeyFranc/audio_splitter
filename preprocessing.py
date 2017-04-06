#!/usr/bin/env python3
import numpy as np
#import scipy.io.wavfile as siow #read/write


def centering(mtx):

    # Special case: uint8
    if mtx.dtype == np.uint8:
        mtx = np.array(mtx*8388608, dtype=np.int16)
    mean = np.mean(mtx)
    centered = np.array(mtx - mean, dtype=mtx.dtype)
    return (mean, centered)

def whitening(mtx):
    X = mtx @ mtx.T
    eigenval, eigenvec = np.linalg.eig(X)
    Drootinv = np.diag(np.sqrt(1/eigenval))
    E = eigenvec
    xtilde = E @ Drootinv @ E.T @ X
    return np.array(xtilde, dtype=mtx.dtype)

