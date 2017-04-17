#!/usr/bin/env python3
import numpy as np
#import scipy.io.wavfile as siow #read/write


def centering(mtx):

    # Get the mean for each channel
    means = [ np.mean(row) for row in mtx.T ]

    # Normalize each channel
    for i in range(len(mtx.T)):
        mtx[:][i] -= means[i]

    # Return normalized channel and mean vectors
    return (means, mtx)

def whitening(mtx):

    # Get the covariance matrix
    X = mtx.T @ mtx
    eigenval, eigenvec = np.linalg.eig(X)
    Drootinv = np.diag(np.sqrt(1/eigenval))
    E = eigenvec
    xtilde = E @ Drootinv @ E.T @ X
    return np.array(xtilde, dtype=mtx.dtype)
