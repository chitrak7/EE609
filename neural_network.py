import numpy as np
#import tensorflow as tf
import sgd
import mbgd
import svrg
import saga
import sag
import mnist
from layer import Layer
import scipy.misc
import matplotlib.pyplot as plt
# to get the image from grayscale values #### scipy.misc.imresize(images[0,:,:] * -1 + 256, (20,20))

def l2(y,yp, der=False):
    if(der):
        return -200*(y-yp)
    return np.dot(y-yp, y-yp)

def cat(y,yp,der=False):
    t = np.amax(yp)
    yp = yp - t
    yp = np.exp(yp)
    t  = np.sum(yp)
    yp =  yp/t
    if(der):
        print(y-yp)
        return (y-yp)
    else:
        return -np.dot(y,np.log(yp))

def softmax(yp,der=False):
    t = np.amax(yp)
    yp = yp - t

    yp = np.exp(yp/100)
    t  = np.sum(yp)
    yp =  yp/t
    return yp
def sigmoid(x, der=False):
    y = np.array([1/(1+np.exp(-i/50)) for i in x])
    if(der):
        y = 0.02*np.multiply(y, 1-y)
    return y
def imgp(x, der=False):
    y = 1-x/255
    if(der):
        y = 0.000025*np.multiply(y, 1-y)
    return y
def relu(x, der=False):
    if(der):
        y = np.array([0 if (i<0) else 0.001 for i in x])
    else:
        y = np.array([0 if (i<0) else 0.001*i for i in x])
    return y

def id(x, der=False):
    if(der):
        return np.array([1 for i in x])
    return x

'''

Implementation of neural network. The neural netwowrk calss uses 6 arguments which are
input dataset "x", input one hot encoded labels  "y", optimizer function "optimizer",
An array specifying the hidden noded, and another specifying activation function with respect
to each node. By default input and output nodes have the size of x and y.
'''

class NN:
    def __init__(self, x, y, hidden_layers, loss=l2, optimizer=sgd.SGD, activation_func=sigmoid):

        self.x                   = x
        self.y                   = y
        self.optimizer           = optimizer
        self.loss                = loss

        if activation_func is not sigmoid:
            self.layers    = [Layer(hidden_layers[0], x.shape[1], activation_func[0])]
            for i in range(len(hidden_layers[1:])):
                self.layers.append(Layer(hidden_layers[i+1], hidden_layers[i], activation_func[i+1]))
            self.layers.append(Layer(y.shape[1], hidden_layers[-1], activation_func[-1]))
        else:
            self.layers    = [Layer(hidden_layers[0], x.shape[1], activation_func)]
            for i in range(len(hidden_layers[1:-1])):
                self.layers.append(Layer(hidden_layers[i+1], hidden_layers[i], activation_func))
            self.layers.append(Layer(y.shape[1], hidden_layers[-1], activation_func))

    def feed_forward(self, test):
        self.layers[0].z          = np.matmul(self.layers[0].weight, test) + self.layers[0].bias
        self.layers[0].activation = self.layers[0].activation_function(self.layers[0].z)

        for i in np.arange(len(self.layers))[1:]:
            self.layers[i].z          = np.matmul(self.layers[i].weight, self.layers[i-1].activation) + self.layers[i].bias
            self.layers[i].activation = self.layers[i].activation_function(self.layers[i].z)

    def predict(self, test):
        self.feed_forward(test)
        return self.layers[-1].activation

    def train(self):
        self.layers = self.optimizer.optimize(self.x, self.y, self.layers, self.loss)

x = np.array([np.random.randint(1, 100, size=(2)) for i in range(4000)])
images = mnist.train_images()
labels =  mnist.train_labels()
img = []
for i in range(len(images)):
    img.append(scipy.misc.imresize(images[i,:,:] * -1 + 256, (20,20)).flatten())
images =   np.array([imgp(i) for i in img])
print(images[1])
y = np.array([1/(i[0]+i[1]) for i in x]).reshape((-1,1))

perm = np.random.permutation(len(images))
images = images[perm]
labels = labels[perm]
labels = [[(1 if (i==j) else 0) for j in range(10)] for i in labels]
images_train =  np.array(images[0:2000])
images_test = np.array(images[2000:2500])
labels_train = np.array(labels[0:2000])
labels_test =  np.array(labels[2000:2500])
x =  images_train
y =  labels_train
optimizer = sgd.SGD()
for i in range(60):
    print(images_train[i]-images_train[i+1])
net = NN(x,y,[100],cat,optimizer,[relu, softmax])
for layer in net.layers:
    print(layer.weight.shape)
net.train()
ct=0
print(net.layers[-1].bias)
print(net.layers[-1].weight)
print(net.layers[-2].activation)
for i in range(500):
    lab1 = np.argmax(net.predict(images_test[i]))
    lab2 = np.argmax(labels_test[i])
    #print(net.layers[-2].activation)
    #plt.imshow(images_test[i].resize(20,20))
    #print(images_test[i])
    if (lab1==lab2):
        ct += 1
print(ct/5)
