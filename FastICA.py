#!/usr/bin/env python3
import numpy as np
import preprocessing as prep
import functions as f
import inout

def one_unit_ica(X, g, dg, w_prev = None):

    w = np.random.random(X.shape[1])
    w_old = np.zeros(w.shape)
    niter = 1000
    iter = 0
    for iter in range(niter):
        exp1 = sum(x * g(np.dot(w,x)) for x in X)/len(X)
        exp2 = sum(dg(np.dot(w,x)) for x in X)/len(X) * w

        w = exp1 - exp2


        #If we are doing multi_unit)_ica, need to decorrelate the vectors
        if w_prev != None:
            assert(type(w_prev == list))
            w -= sum([np.dot(w.T, wj) * wj for wj in w_prev])

        #normalize
        w = w/np.linalg.norm(w)

        if np.allclose(abs(np.dot(w, w_old)), 1):
            break
        else:
            w_old = w


    return w

def multi_unit_ica(X, g, dg, num_components):
    W = [0]*num_components #'empty' list to hold each weight vector
    W[0] = one_unit_ica(X, g, dg)

    for n in range(1, num_components):
        W[n] = one_unit_ica(X, g, dg, W[:n]) #independent component

    return W



def trial():
    S = inout.DGP()
    S.set_sources()
    S = S.get_mixed()

    num_components = S.shape[1]
    mean, centered = prep.centering(S)
    whitened = prep.whitening(centered)

    W = np.array(multi_unit_ica(S, f.dexp, f.d2exp, num_components))
    return W, S


def ica(fn):
# Runs ica on a file named "fn"

    # Get input as an array for each channel
    io = inout.WavIO(fn)
    channels = io.read_source()
    
    # First, preprocess data
    means, centered = prep.centering(channels)
    processed = prep.whitening(centered)
    print(len(processed))

    # Second, run ICA
    #Note: Using first and second derivatives of G
    W = multi_unit_ica(processed, f.exp, f.dexp, 3)

    # Third, add the means back
    for i in range(len(means)):
        centered[:][i] += means[i]

    # Fourth, get sources from sample, adding back the means
    S = W*processed

    # Finally, Write output files
    io.write_sources(S)

    

if __name__ == '__main__':

    ica('test.wav')

