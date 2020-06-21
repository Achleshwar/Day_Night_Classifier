import numpy as np 
import sys
sys.path.append('../activation_function/')
from activation_function import relu , sigmoid

"""
The linear forward module (vectorized over all the examples) computes the following equations:

                                      𝑍[𝑙]=𝑊[𝑙]𝐴[𝑙−1]+𝑏[𝑙]
where 𝐴[0]=𝑋 .
"""

def linear_forward(A,W,b):
    """Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently"""
    
    Z = np.dot(W,A) + b
    
    cache = (A,W,b)
    
    return Z, cache

"""
 Implement the forward propagation of the LINEAR->ACTIVATION layer.

                    Mathematical relation is:  𝐴[𝑙]=𝑔(𝑍[𝑙])=𝑔(𝑊[𝑙]𝐴[𝑙−1]+𝑏[𝑙])  
where the activation "g" can be sigmoid() or relu()
"""

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    if(activation == 'relu'):
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        
    elif(activation == 'sigmoid'):
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
        
    cache = (linear_cache, activation_cache)
    
    return A, cache
        