#import tensorflow as tf
import numpy      as np
from layer import Layer

def lr(i):
    return np.power(1+i, -0.7)

def feed_forward(layers, test):
    layers[0].z          = np.matmul(layers[0].weight, test) + layers[0].bias
    layers[0].activation = layers[0].activation_function(layers[0].z)

    for i in np.arange(len(layers))[1:]:
        layers[i].z          = np.matmul(layers[i].weight, layers[i-1].activation) + layers[i].bias
        layers[i].activation = layers[i].activation_function(layers[i].z)

class SAGA:
    def __init__(self, lr=lr, niter=1000):
        self.learning_rate = lr
        self.niter         = niter

    def optimize(self, x, y, layers, loss):
        n = x.shape[0]
        layersx = []
        layersk = []
        for j in range(len(layers)):
            layersk.append(Layer(layers[j].size, layers[j].size_l, layers[j].activation_function))
        for i in range(n):
            layerst  = layers
            feed_forward(layers, x[i])
            layerst[-1].bias_g =  np.multiply(loss(y[i], layers[-1].activation, der=True), layers[-1].activation_function(layers[-1].z, der=True))
            layerst[-1].weight_g = np.matmul(np.reshape(layers[-2].activation, (-1,1)), np.reshape(layers[-1].bias_g, (1,-1))).T
            layersk[-1].bias_g += layerst[-1].bias_g
            layersk[-1].weight_g += layerst[-1].weight_g
            for k in np.arange(len(layers)-2,0,-1):
                layerst[k].bias_g = np.multiply(np.matmul(layers[k+1].weight.T, layers[k+1].bias_g), layers[k].activation_function(layers[k].z, der=True))
                layerst[k].weight_g = np.matmul(np.reshape(layers[k-1].activation, (-1,1)), np.reshape(layers[k].bias_g, (1,-1))).T
                layersk[k].bias_g += layerst[k].bias_g
                layersk[k].weight_g += layerst[k].weight_g
            layerst[0].bias_g = np.multiply(np.matmul(layers[1].weight.T, layers[1].bias_g), layers[0].activation_function(layers[0].z, der=True))
            layerst[0].weight_g = np.matmul(np.reshape(x[i], (-1,1)), np.reshape(layers[0].bias_g, (1,-1))).T
            layersk[0].bias_g += layerst[0].bias_g
            layersk[0].weight_g += layerst[0].weight_g
            layersx.append(layerst)

        for j in range(len(layers)):
            layersk[j].bias_g = (1/n)*layersk[j].bias_g
            layersk[j].weight_g = (1/n)*layersk[j].weight_g
        for iter in range(self.niter):
            layersp = []
            lr = self.learning_rate(iter)
            i = np.random.randint(n)
            for j in range(len(layers)):
                layersp.append(Layer(layers[j].size, layers[j].size_l, layers[j].activation_function))
            layersp[-1].bias_g =  (1/n)*np.multiply(loss(y[i], layers[-1].activation, der=True), layers[-1].activation_function(layers[-1].z, der=True))
            layersp[-1].weight_g = (1/n)*np.matmul(np.reshape(layers[-2].activation, (-1,1)), np.reshape(layers[-1].bias_g, (1,-1))).T
            layers[-1].weight -= lr*layersk[-1].weight_g
            layers[-1].bias   -= lr*layersk[-1].bias_g
            layers[-1].bias   -= lr*(layersp[-1].bias_g - layersx[i][-1].bias_g)
            layers[-1].weight   -= lr*(layersp[-1].weight_g - layersx[i][-1].weight_g)
            layersk[-1].bias_g -= layersx[i][-1].bias_g*(1/n)
            layersk[-1].weight_g -= layersx[i][-1].weight_g*(1/n)
            layersx[i][-1].bias_g = layersp[-1].bias_g
            layersx[i][-1].weight_g = layersp[-1].weight_g
            layersk[-1].bias_g += layersx[i][-1].bias_g*(1/n)
            layersk[-1].weight_g += layersx[i][-1].weight_g*(1/n)

            for k in np.arange(len(layers)-2,0,-1):
                layersp[k].bias_g = np.multiply(np.matmul(layers[k+1].weight.T, layers[k+1].bias_g), layers[k].activation_function(layers[k].z, der=True))
                layersp[k].weight_g = np.matmul(np.reshape(layers[k-1].activation, (-1,1)), np.reshape(layers[k].bias_g, (1,-1))).T
                layers[k].weight -= lr*layersk[k].weight_g
                layers[k].bias   -= lr*layersk[k].bias_g
                layers[k].bias   -= lr*(layersp[k].bias_g - layersx[i][k].bias_g)
                layers[k].weight   -= lr*(layersp[k].weight_g - layersx[i][k].weight_g)
                layersk[k].bias_g -= layersx[i][k].bias_g*(1/n)
                layersk[k].weight_g -= layersx[i][k].weight_g*(1/n)
                layersx[i][k].bias_g = layersp[k].bias_g
                layersx[i][k].weight_g = layersp[k].weight_g
                layersk[k].bias_g += layersx[i][k].bias_g*(1/n)
                layersk[k].weight_g += layersx[i][k].weight_g*(1/n)

            layersp[0].bias_g = np.multiply(np.matmul(layers[1].weight.T, layers[1].bias_g), layers[0].activation_function(layers[0].z, der=True))
            layersp[0].weight_g = np.matmul(np.reshape(x[i], (-1,1)), np.reshape(layers[0].bias_g, (1,-1))).T
            layers[0].weight -= lr*layersk[0].weight_g
            layers[0].bias   -= lr*layersk[0].bias_g
            layers[0].bias   -= lr*(layersp[0].bias_g - layersx[i][0].bias_g)
            layers[0].weight   -= lr*(layersp[0].weight_g - layersx[i][0].weight_g)
            layersk[0].bias_g -= layersx[i][0].bias_g*(1/n)
            layersk[0].weight_g -= layersx[i][0].weight_g*(1/n)
            layersx[i][0].bias_g = layersp[0].bias_g
            layersx[i][0].weight_g = layersp[0].weight_g
            layersk[0].bias_g += layersx[i][0].bias_g*(1/n)
            layersk[0].weight_g += layersx[i][0].weight_g*(1/n)


            '''
            for j in range(len(layers)):
                print("activation", layers[j].activation)
                print("z", layers[j].z)
                print("bias", layers[j].bias)
                print("weight",layers[j].weight)
            '''

        return layers
