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
        w /=np.linalg.norm(w)

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
    import matplotlib.pyplot as plt

    S = inout.DGP()
    S.set_sources()
    X = S.get_mixed()

    num_components = X.shape[1]
    mean, centered = prep.centering(X)
    whitened = prep.whitening(centered)

    W = np.array(multi_unit_ica(X, f.dexp, f.d2exp, num_components))

    model = X @ W
    colors = ['red', 'steelblue','orange']
    plt.axhline(0, color="black")
    plt.title("Original Signal")
    for signal, color in  zip(S.sources.T, colors):
        plt.plot(signal, color=color)
    plt.show()
    plt.axhline(0, color="black")
    plt.title("Recovered Signal")
    for signal, color in  zip(model.T, colors):
        plt.plot(signal, color=color)
    plt.show()
    return W, X


def ica(fn):
# Runs ica on a file named "fn"

    # Get input as an array for each channel
    io = inout.WavIO(fn)
    channels = io.read_source()
    
    # Now perform the algorithm seperately on each channel
    sources = []
    for channel in channels:

        # First, preprocess data
        mean, centered = prep.centering(channel)
        processed = prep.whitening(centered)
    
        # Second, run ICA
        #Note: Using first and second derivatives of G
        sources += [multi_unit_ica(processed, f.dexp, f.d2exp, 3)]

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

    ica('elvis_riverside.wav')
