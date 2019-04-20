import numpy as np

'''
Layers for Neural Network.
Constructs a neural network layer of size "size" which is on the top of a Neural
Netwok of size "size_l" with activation fuction "activation_function". 
'''


class Layer:
    def __init__(self, size, size_l, activation_function):
        self.size                = size
        self.size_l              = size_l
        self.activation_function =  activation_function
        self.z                   = np.zeros(size)
        self.activation          = activation_function(self.z)
        self.weight              = np.random.rand(size, size_l)
        self.bias                = np.random.rand(size)
        self.weight_g            = np.zeros(self.weight.shape)
        self.bias_g              = np.zeros(self.bias.shape)
