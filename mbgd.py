#import tensorflow as tf
import numpy      as np
from layer import Layer


def lr(i):
    return 10

def feed_forward(layers, test):
    layers[0].z          = np.matmul(layers[0].weight, test) + layers[0].bias
    layers[0].activation = layers[0].activation_function(layers[0].z)

    for i in np.arange(len(layers))[1:]:
        layers[i].z          = np.matmul(layers[i].weight, layers[i-1].activation) + layers[i].bias
        layers[i].activation = layers[i].activation_function(layers[i].z)

class MBGD:
    def __init__(self, lr=lr, niter=100, batch_size=100):
        self.learning_rate = lr
        self.niter         = niter
        self.batch_size    = batch_size

    def optimize(self, x, y, layers, loss):
        n = x.shape[0]
        print(self.niter)
        for iter in range(self.niter):
            print(iter)
            i = np.random.randint(n,size=(self.batch_size))
            lr = self.learning_rate(iter)/self.batch_size

            '''
            for j in range(len(layers)):
                print("activation", layers[j].activation)
                print("z", layers[j].z)
                print("bias", layers[j].bias)
                print("weight",layers[j].weight)
            '''
            layers[-1].bias_g =  np.zeros(layers[-1].bias.shape)
            layers[-1].weight_g = np.zeros(layers[-1].weight.shape)
            layersk = []
            for j in range(len(layers)):
                layersk.append(Layer(layers[j].size, layers[j].size_l, layers[j].activation_function))

            for ll in range(self.batch_size):
                feed_forward(layers, x[i[ll]])
                layers[-1].bias_g =  0.01*(layers[-1].activation - y[i[ll]])
                layers[-1].weight_g = np.matmul(np.reshape(layers[-2].activation, (-1,1)), np.reshape(layers[-1].bias_g, (1,-1))).T
                layersk[-1].bias_g += layers[-1].bias_g
                layersk[-1].weight_g += layers[-1].weight_g
            layers[-1].weight -= lr*layersk[-1].weight_g
            layers[-1].bias   -= lr*layersk[-1].bias_g
            for k in np.arange(len(layers)-2,0,-1):
                layers[k].bias_g =  np.zeros(layers[k].bias.shape)
                layers[k].weight_g = np.zeros(layers[k].weight.shape)
                for ll in range(self.batch_size):
                    feed_forward(layers, x[i[ll]])
                    layers[k].bias_g = np.multiply(np.matmul(layers[k+1].weight.T, layers[k+1].bias_g), layers[k].activation_function(layers[k].z, der=True))
                    layers[k].weight_g = np.matmul(np.reshape(layers[k-1].activation, (-1,1)), np.reshape(layers[k].bias_g, (1,-1))).T
                    layersk[-1].bias_g += layers[-1].bias_g
                    layersk[-1].weight_g += layers[-1].weight_g
                layers[k].weight -= lr*layersk[k].weight_g
                layers[k].bias   -= lr*layersk[k].bias_g

            layers[0].bias_g =  np.zeros(layers[0].bias.shape)
            layers[0].weight_g = np.zeros(layers[0].weight.shape)
            for ll in range(self.batch_size):
                feed_forward(layers, x[i[ll]])
                layers[0].bias_g += np.multiply(np.matmul(layers[1].weight.T, layers[1].bias_g), layers[0].activation_function(layers[0].z, der=True))
                layers[0].weight_g += np.matmul(np.reshape(x[i[ll]], (-1,1)), np.reshape(layers[0].bias_g, (1,-1))).T
                layersk[0].bias_g += layers[0].bias_g
                layersk[0].weight_g += layers[0].weight_g
            layers[0].weight -= lr*layersk[0].weight_g
            layers[0].bias   -= lr*layersk[0].bias_g
            '''
            for j in range(len(layers)):
                print("bias_g", layers[j].bias_g)
                print("weight_g", layers[j].weight_g)
            '''
        return layers
