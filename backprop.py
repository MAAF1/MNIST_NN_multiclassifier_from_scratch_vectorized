import numpy as np

def backward(X_batch, y_batch, W3, W2, out1, out2, out3):
    dE_dnet3 = out3 - y_batch # 32 x 10
    dE_dout2 = np.dot(dE_dnet3, W3.T) # 32 x 15  
    dE_dnet2 = (dE_dout2 * (1 - (out2 ** 2))) # 32 x 15
    dE_dout1 = np.dot(dE_dnet2, W2.T)# 32 x 20
    dE_dnet1 = (dE_dout1 * (1 - (out1 ** 2))) # 32 x 20


    dW3 = np.dot(out2.T, dE_dnet3)
    db3 = np.sum(dE_dnet3,axis = 0, keepdims=True)
    dW2 = np.dot(out1.T, dE_dnet2)
    db2 = np.sum(dE_dnet2,axis=0,keepdims=True)
    dW1 = np.dot(X_batch.T, dE_dnet1)
    db1 = np.sum(dE_dnet1,axis=0,keepdims=True)
    return dW1, db1, dW2, db2, dW3, db3


