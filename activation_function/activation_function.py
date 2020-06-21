import numpy as np
"""
Sigmoid:
        𝜎(𝑍)=𝜎(𝑊𝐴+𝑏)=1 / (1+𝑒xp−(𝑊𝐴+𝑏))
"""
def sigmoid(Z):
    """ Implement sigmoid function as described above
    
    Input:
    Z --- Output of the linear_forward function
    
    Returns:
    A --- activation value (input for next layer)
    activation_cache ---- Z (this will be used while implementing backpropagation)
    """
    
    A = 1 / (1 + np.exp(-Z))
    
    activation_cache = Z
    
    return A, activation_cache

def sigmoid_backward(dA, activation_cache):
    """
    Implement sigmoid_backward for backpropagation
    
    Input:
    "dA" --- derivative of activation value of current layer w.r.t. cost function
    "activation_cache" --- Z of current layer
    
    Returns:
    "dZ" --- derivative of Z w.r.t. cost function
    """
    
    A, _ = sigmoid(activation_cache)
    dg = np.multiply(A, 1-A)
    
    dZ = np.multiply(dA, dg)
    
    return dZ

"""
 ReLU:
  The mathematical formula for ReLu is  𝐴=𝑅𝐸𝐿𝑈(𝑍)=𝑚𝑎𝑥(0,𝑍) 
 """

def relu(Z):
    """ Implement relu function as described above
    
    Input:
    Z --- Output of the linear_forward function
    
    Returns:
    A --- activation value (input for next layer)
    activation_cache ---- Z (this will be used while implementing backpropagation)
    """
    
    A = np.where(Z > 0 , Z , 0)
    
    activation_cache = Z
    
    return A, activation_cache

def relu_backward(dA, activation_cache):
    """
    Implement sigmoid_backward for backpropagation
    
    Input:
    "dA" --- derivative of activation value of current layer w.r.t. cost function
    "activation_cache" --- Z of current layer
    
    Returns:
    "dZ" --- derivative of Z w.r.t. cost function
    """
    
    dg = np.where(activation_cache >=0, 1, 0)
    
    dZ = np.multiply(dA, dg)
    
    return dZ