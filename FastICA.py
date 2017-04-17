#!/usr/bin/env python3
import numpy as np
import preprocessing as prep
import functions as f
import inout

def one_unit_ica(X, g, dg):

    w = np.random.random(X.shape[1])
    w_old = np.zeros(w.shape)
    niter = 1000
    iter = 0
    for iter in range(niter):
        exp1 = sum(x * g(np.dot(w,x)) for x in X)/len(X)
        exp2 = sum(dg(np.dot(w,x)) for x in X)/len(X) * w

        w = exp1 - exp2
        w = w/np.linalg.norm(w)

        if np.allclose(abs(np.dot(w, w_old)), 1):
            break
        else:
            w_old = w


    return w

def multi_unit_ica(X, g, dg, num_components):
    W = [0]*num_components #'empty' list to hold each weight vector
    for n in range(num_components):
        W[n] = one_unit_ica(X, g, dg) #independent component
        W[n] = W[n] - sum(np.dot(W[n], w)*w for w in W[:n]) #decorrelate
        W[n] = W[n] / np.linalg.norm(W[n])

    return W

def ica(fn):
# Runs ica on a file named "fn"

    # Get input as an array for each channel
    io = inout.WavIO(fn)
    channels = io.read_source()
    
    # First, preprocess data
    means, centered = prep.centering(channels)
    processed = prep.whitening(centered)

    # Second, run ICA
    W = [multi_unit_ica(processed, f.exp, f.dexp, 3)]

    # Concatenate channels
    for channel in sources[1:]:
        i=0
        for source in channel:
            sources[0][i] = np.append(sources[0][i], source, axis=1)
            i+=1

    # Add mean back
    sources[0] += mean

    # Write output files
    io.write_sources(sources[0])

    

if __name__ == '__main__':

    ica('test.wav')

