#import tensorflow as tf
import numpy      as np
from layer import Layer
import matplotlib.pyplot as plt
def lr(i):
    return 10

def feed_forward(layers, test):
    layers[0].z          = np.matmul(layers[0].weight, test) + layers[0].bias
    layers[0].activation = layers[0].activation_function(layers[0].z)

    for i in np.arange(len(layers))[1:]:
        layers[i].z          = np.matmul(layers[i].weight, layers[i-1].activation) + layers[i].bias
        layers[i].activation = layers[i].activation_function(layers[i].z)

class SVRG:
    def __init__(self, lr=lr, niter=700, batch_size=1000):
        self.learning_rate = lr
        self.niter         = niter
        self.batch_size    = batch_size

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
            print(iter)
            layersp = layers
            layersk = []
            for j in range(len(layers)):
                layersk.append(Layer(layers[j].size, layers[j].size_l, layers[j].activation_function))

            for i in range(n):
                feed_forward(layersp, x[i])
                layers[-1].bias_g =  0.01*(layersp[-1].activation - y[i])
                layers[-1].weight_g = np.matmul(np.reshape(layersp[-2].activation, (-1,1)), np.reshape(layersp[-1].bias_g, (1,-1))).T
                layersk[-1].bias_g += layers[-1].bias_g
                layersk[-1].weight_g += layers[-1].weight_g
                for k in np.arange(len(layers)-2,0,-1):
                    layers[k].bias_g = np.multiply(np.matmul(layersp[k+1].weight.T, layersp[k+1].bias_g), layersp[k].activation_function(layersp[k].z, der=True))
                    layers[k].weight_g = np.matmul(np.reshape(layersp[k-1].activation, (-1,1)), np.reshape(layersp[k].bias_g, (1,-1))).T
                    layersk[k].bias_g += layers[k].bias_g
                    layersk[k].weight_g += layers[k].weight_g
                layers[0].bias_g = np.multiply(np.matmul(layersp[1].weight.T, layersp[1].bias_g), layersp[0].activation_function(layersp[0].z, der=True))
                layers[0].weight_g = np.matmul(np.reshape(x[i], (-1,1)), np.reshape(layersp[0].bias_g, (1,-1))).T
                layersk[0].bias_g += layers[0].bias_g
                layersk[0].weight_g += layers[0].weight_g
            for j in range(len(layers)):
                layersk[j].weight_g = (1/n)*layersk[j].weight_g
                layersk[j].bias_g = (1/n)*layersk[j].bias_g
            for ll in range(self.batch_size):
                i = np.random.randint(n)
                lr = self.learning_rate(iter)
                feed_forward(layers, x[i])
                feed_forward(layersp, x[i])
                layersm = []
                for j in range(len(layers)):
                    layersm.append(Layer(layers[j].size, layers[j].size_l, layers[j].activation_function))
                layers[-1].bias_g =  0.01*(layers[-1].activation - y[i])
                layers[-1].weight_g = np.matmul(np.reshape(layers[-2].activation, (-1,1)), np.reshape(layers[-1].bias_g, (1,-1))).T
                layersm[-1].bias_g = 0.01*(layersp[-1].activation - y[i])
                layersm[-1].weight_g = np.matmul(np.reshape(layersp[-2].activation, (-1,1)), np.reshape(0.01*(layersp[-1].activation - y[i]), (1,-1))).T
                layers[-1].bias   -= lr*layersk[-1].bias_g
                layers[-1].weight -= lr*layersk[-1].weight_g
                layers[-1].weight -= lr*layers[-1].weight_g
                layers[-1].bias   -= lr*layers[-1].bias_g
                layers[-1].bias   += lr*layersm[-1].bias_g
                layers[-1].weight += lr*layersm[-1].weight_g
                for k in np.arange(len(layers)-2,0,-1):
                    layers[k].bias_g = np.multiply(np.matmul(layers[k+1].weight.T, layers[k+1].bias_g), layers[k].activation_function(layers[k].z, der=True))
                    layers[k].weight_g = np.matmul(np.reshape(layers[k-1].activation, (-1,1)), np.reshape(layers[k].bias_g, (1,-1))).T
                    layersm[k].bias_g = np.multiply(np.matmul(layersp[k+1].weight.T, layersm[k+1].bias_g), layersp[k].activation_function(layersp[k].z, der=True))
                    layersm[k].weight_g = np.matmul(np.reshape(layersp[k-1].activation, (-1,1)), np.reshape(layersm[k].bias_g, (1,-1))).T
                    layers[k].bias   -= lr*layersk[k].bias_g
                    layers[k].weight -= lr*layersk[k].weight_g
                    layers[k].weight -= lr*layers[k].weight_g
                    layers[k].bias   -= lr*layers[k].bias_g
                    layers[k].bias   += lr*layersm[k].bias_g
                    layers[k].weight += lr*layersm[k].weight_g
                layers[0].bias_g = np.multiply(np.matmul(layers[1].weight.T, layers[1].bias_g), layers[0].activation_function(layers[0].z, der=True))
                layers[0].weight_g = np.matmul(np.reshape(x[i], (-1,1)), np.reshape(layers[0].bias_g, (1,-1))).T
                layersm[0].bias_g = np.multiply(np.matmul(layersp[1].weight.T, layersm[1].bias_g), layersp[0].activation_function(layersp[0].z, der=True))
                layersm[0].weight_g = np.matmul(np.reshape(x[i], (-1,1)), np.reshape(layersm[0].bias_g, (1,-1))).T
                layers[0].bias   -= lr*layersk[0].bias_g
                layers[0].weight -= lr*layersk[0].weight_g
                layers[0].weight -= lr*layers[0].weight_g
                layers[0].bias   -= lr*layers[0].bias_g
                layers[0].bias   += lr*layersm[0].bias_g
                layers[0].weight += lr*layersm[0].weight_g
            grad2 = 0
            ct=0
            '''
            Evaluate trainig loss and square of norm of Gradient for every interaion
            '''
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
            xt.append(iter*(x.shape[0]+self.batch_size)/x.shape[0])
            acc.append(ct/x.shape[0])
            grd.append(grad2)


        plt.plot(xt,acc)
        plt.title('Training loss Plot SVRG')
        plt.xlabel('# Grad/(n+m)')
        plt.ylabel('Training loss')
        plt.show()
        plt.clf()
        plt.plot(xt,grd)
        plt.title('Gradient norm Plot SVRG')
        plt.xlabel('# Grad/(n+m)')
        plt.ylabel('||grad(f(x))||^2')
        plt.show()
        return layers
