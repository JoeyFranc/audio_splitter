#!/usr/bin/env python3
import numpy as np
#import scipy.io.wavfile as siow #read/write


def centering(mtx):

    unsigned = False

    # Special case: uint8
    if mtx.dtype == np.uint8:
        mtx = np.array(mtx, dtype=np.int16)
        unsigned = True
    mean = np.mean(mtx)
    mtx -= mean

    # Convert special case to int8
    if unsigned:
        mtx = np.array(mtx, dtype=np.int8)
    return (mean, centered)

def whitening(mtx):
    X = mtx @ mtx.T
    eigenval, eigenvec = np.linalg.eig(X)
    Drootinv = np.diag(np.sqrt(1/eigenval))
    E = eigenvec
    xtilde = E @ Drootinv @ E.T @ X
    return np.array(xtilde, dtype=mtx.dtype)
