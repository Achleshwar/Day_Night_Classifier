import numpy as np
import math

def random_mini_batches(X,y,mini_batch_size, seed):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for night / 0 for day), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- (mini_batch_X, mini_batch_Y)
    """
    
    np.random.seed(seed)
    m = X.shape[1] #no of examples
    mini_batches = []
    
    #Shuffle X and y
    indices = list(np.random.permutation(m))
    shuffled_X = X[:, indices]
    shuffled_y = y[:, indices].reshape((1,m))
    
    #create mini_batches , minus the end case
    total_mini_batches = math.floor(m/mini_batch_size) #this doesn't include end case if m is not multiple of mini_batch_size
    for k in range(total_mini_batches):
        mini_batch_X = shuffled_X[:, k*mini_batch_size: (k+1)*mini_batch_size]
        mini_batch_y = shuffled_y[:, k*mini_batch_size: (k+1)*mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_y)
        mini_batches.append(mini_batch)
        
    #handle the end case
    if(m%mini_batch_size !=0):
        mini_batch_X = shuffled_X[:, total_mini_batches*mini_batch_size:]
        mini_batch_y = shuffled_y[:, total_mini_batches*mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_y)
        mini_batches.append(mini_batch_X, mini_batch_y)
        
    return mini_batches