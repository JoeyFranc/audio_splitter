#!/usr/bin/env python3
import numpy as np
#import scipy.io.wavfile as siow #read/write


def centering(mtx):
    mean = np.mean(mtx)
    centered = mtx - mean
    return centered

def whitening(mtx):
    X = mtx @ mtx.T
    eigenval, eigenvec = np.linalg.eig(X)
    Drootinv = np.diag(np.sqrt(1/eigenval))
    E = eigenvec
    xtilde = E @ Drootinv @ E.T @ X
    return xtilde

