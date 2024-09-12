import numpy as np
import os

def softmax_vectorized(X):
    denominator = np.sum(np.exp(X),axis = 1,keepdims=True)

    return  np.exp(X)/denominator
