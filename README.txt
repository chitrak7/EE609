Enviornment:
Python3.6
Numpy
matplotlib
mnist


How to run:
1> Install all the dependencies.
2> Set up the optimiser. Optimisers implemented in this code are SGD,  MBGD, SVRG, SAG, SAGA
optimiser = sgd.SGD(niter=10000)
3> Set up the neural network. An exaple neural network is given below
net = NN(x,y,[100],cat,optimizer,[relu, softmax])
5> Tweak activation functions if required.
4> execute python3 neural_net.py


Important:
1> Ensure that activation function support "der" call which when provided True
   returns the derivtive instead of activation
    eg:
    activation_func(x, der):
      if(der):
        return f'(x)
      else
        return f(x)

2> The code is only for categorical output layer. 
