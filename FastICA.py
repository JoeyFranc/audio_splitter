#!/usr/bin/env python3
import numpy as np
import preprocessing
import functions

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
