import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def ReLU(x):
    return np.maximum(x, 0)

def ReLU_prime(x):
    return x > 0

def softmax(x):
    A = np.exp(x) / np.sum(np.exp(x))
