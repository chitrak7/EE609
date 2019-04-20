#import tensorflow as tf
import numpy      as np
from layer import Layer
import matplotlib.pyplot as plt

def lr(i):
    return 100

def feed_forward(layers, test):
    layers[0].z          = np.matmul(layers[0].weight, test) + layers[0].bias
    #print(layers[0].z)
    layers[0].activation = layers[0].activation_function(layers[0].z)

    for i in np.arange(len(layers))[1:]:
        layers[i].z          = np.matmul(layers[i].weight, layers[i-1].activation) + layers[i].bias
        layers[i].activation = layers[i].activation_function(layers[i].z)

    #print(layers[-1].z)
    #print(layers[-1].activation)
class SGD:
    def __init__(self, lr=lr, niter=2100000):
        self.learning_rate = lr
        self.niter         = niter

    def optimize(self, x, y, layers, loss):
        n = x.shape[0]
        print(self.niter)
        xt = []
        grd = []
        acc = []
        for iter in range(self.niter):
            '''
            backpropagation code for every iteration
            '''
            i = np.random.randint(n)
            #print(y[i])
            lr = self.learning_rate(iter)
            feed_forward(layers, x[i])


            layers[-1].bias_g =  0.01*(layers[-1].activation - y[i])
            layers[-1].weight_g = np.matmul(np.reshape(layers[-1].bias_g, (-1,1)), np.reshape(layers[-2].activation, (1,-1)))
            layers[-1].weight -= lr*layers[-1].weight_g
            layers[-1].bias   -= lr*layers[-1].bias_g
            for k in np.arange(len(layers)-2,0,-1):
                layers[k].bias_g = np.multiply(np.matmul(layers[k+1].weight.T, layers[k+1].bias_g), layers[k].activation_function(layers[k].z, der=True))
                layers[k].weight_g = np.matmul(np.reshape(layers[k-1].activation, (-1,1)), np.reshape(layers[k].bias_g, (1,-1))).T

                layers[k].weight -= lr*layers[k].weight_g
                layers[k].bias   -= lr*layers[k].bias_g
            layers[0].bias_g = np.multiply(np.matmul(layers[1].weight.T, layers[1].bias_g), layers[0].activation_function(layers[0].z, der=True))
            layers[0].weight_g = np.matmul(np.reshape(x[i], (-1,1)), np.reshape(layers[0].bias_g, (1,-1))).T

            layers[0].weight -= lr*layers[0].weight_g
            layers[0].bias   -= lr*layers[0].bias_g
            feed_forward(layers,x[i])
            '''
            Evaluate trainig loss and square of norm of Gradient for every 2000 interaions
            '''
            if(iter%2000==0):
                print(iter/2000)
                grad2 = 0
                ct=0
                for zz in range(x.shape[0]):
                    feed_forward(layers, x[zz])

                    ct -= np.dot(y[zz], np.log(layers[-1].activation))

                    layers[-1].bias_g =  0.01*(layers[-1].activation - y[zz])
                    layers[-1].weight_g = np.matmul(np.reshape(layers[-1].bias_g, (-1,1)), np.reshape(layers[-2].activation, (1,-1)))
                    grad2 += np.dot(layers[-1].bias_g,layers[-1].bias_g)
                    grad2 += np.dot(layers[-1].weight_g.flatten(),layers[-1].weight_g.flatten())
                    for k in np.arange(len(layers)-2,0,-1):
                        layers[k].bias_g = np.multiply(np.matmul(layers[k+1].weight.T, layers[k+1].bias_g), layers[k].activation_function(layers[k].z, der=True))
                        layers[k].weight_g = np.matmul(np.reshape(layers[k-1].activation, (-1,1)), np.reshape(layers[k].bias_g, (1,-1))).T
                        grad2 += np.dot(layers[k].bias_g,layers[k].bias_g)
                        grad2 += np.dot(layers[k].weight_g.flatten(),layers[k].weight_g.flatten())
                    layers[0].bias_g = np.multiply(np.matmul(layers[1].weight.T, layers[1].bias_g), layers[0].activation_function(layers[0].z, der=True))
                    layers[0].weight_g = np.matmul(np.reshape(x[zz], (-1,1)), np.reshape(layers[0].bias_g, (1,-1))).T
                    grad2 += np.dot(layers[0].bias_g,layers[0].bias_g)
                    grad2 += np.dot(layers[0].weight_g.flatten(),layers[0].weight_g.flatten())
                print(ct, grad2)
                xt.append(iter/x.shape[0])
                acc.append(ct/x.shape[0])
                grd.append(grad2)

        plt.plot(xt,acc)
        plt.title('Training loss Plot SGD')
        plt.xlabel('# Grad/(n+m)')
        plt.ylabel('Training loss')
        plt.show()
        plt.clf()
        plt.plot(xt,grd)
        plt.title('Gradient norm Plot SGD')
        plt.xlabel('# Grad/(n+m)')
        plt.ylabel('||grad(f(x))||^2')
        plt.show()
        return layers
