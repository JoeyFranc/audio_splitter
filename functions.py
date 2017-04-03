#!/usr/bin/env python3
import numpy as np

def cosh(u, a = 1):
    return (1/a) * np.log(np.cosh(a * u))

def dcosh(u, a = 1):
    return np.tanh(a * u)

def exp(u):
    return -np.exp((-1/2) * u**2)

def dexp(u):
    return -u * exp(u)

def negentropy(u, g = exp):
    '''Honestly looks like this isn't even used in practice,
    according to scipy docs'''
    raise ValueError
