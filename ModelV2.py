# This version 2 of the model will implement a more modular approach.

# Goal is to split the neural network into different anaylyzable parts
# Will consist of layers, activations, losses, and optimizers

# Ideas for layers -- dense layers, sparse layers
# Ideas for activations -- sigmoid, relu, softmax, linear, tanh
# Ideas for losses -- Cat-Crossentropy
# Ideas for optimizers -- hardest part -- SGD, Adam

# What will hold the input value? Will that be an input layer

import numpy as np
import math


class Layer:
    def __init__(self, params: (int, int)):
        # Layers will have an array of neurons it is keeping track of
        # Will have a number of in and out channels
        self.num_in = params[0]
        self.num_out = params[1]
        self.shape = params
        # these are the actual weights of this layer, based on the number of in channels and number of out channels
        lower, upper = -(1.0 / math.sqrt(self.num_in)
                         ), (1.0/math.sqrt(self.num_in))

        self.weight = [[lower + np.random.random() * (upper-lower) for i in range(params[1])]
                       for j in range(params[0])]

    def __str__(self):
        s = ""
        for i in range(self.shape[0]):
            s += str(self.weight[i]) + "\n"
        return s

    def __repr__(self):
        return "Layer with dims (" + str(self.num_in) + "," + str(self.num_out)+")"


class Input:
    def __init__(self, params: int):
        self.num_inputs = params


class Dense(Layer):
    def __init__(self, params: (int, int)):
        # Dense layer for neural netowrk
        super(Dense, self).__init__(params)

    def Mul(self, other: Layer) -> Layer:
        # Returns a dot product with next Layer
        # if the shape of the other layer is not the same as the shape of this layer: return that there is a problem
        if (other.shape[0] != self.shape[1]):
            # This is a problem, as matrix multiplication cannot be performed
            raise BaseException("Argument 1 dim.0 of size '" +
                                str(other.shape[0]) + "' not operable with Argument 0 dim.1 of size '" + str(self.shape[1]) + "'")
        # Matrix multiplication can be performed
        # a matrix with coords
        # c1 will be a mat with dims. [self[0],other[1]]
        t = Dense((self.shape[0], other.shape[1]))
        c = t.weight
        for i in range(len(c)):
            for j in range(len(c[i])):
                a = self.weight[i][0:self.shape[1]]
                b = [other.weight[k][j] for k in range(0, other.shape[0])]
                c[i][j] = 0
                # a and b should always be same length
                for l in range(len(a)):
                    c[i][j] += a[l] * b[l]
        return t

    def __repr__(self):
        return "Dense layer dims. (" + str(self.shape[0]) + "," + str(self.shape[1]) + ")"


class Module:
    def __init__(self):
        self.layers = []
        self.activations = []

    def add(self, obj: Dense):
        # adds a layer to the layers
        self.layers.append(obj)


if __name__ == "__main__":
    np.random.seed(1)
    l1 = Dense((3, 3))
    l2 = Dense((3, 3))
    a = Module()
    a.add(Dense((3, 3)))
    a.add(Dense((3, 8)))

    print(a.layers)
