import numpy as np

from softmax_batch import softmax_vectorized
def forward_batch(X_batch, W1, b1, W2, b2, W3, b3):
    net1 = np.dot(X_batch, W1) + b1 
    out1 = np.tanh(net1)
    net2 = np.dot(out1, W2) + b2
    out2 = np.tanh(net2)
    net3 = np.dot(out2, W3) + b3 
    out3 = softmax_vectorized(net3)
    return  out1, out2, out3

