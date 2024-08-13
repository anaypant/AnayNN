import numpy as np
import math

activationSet = set()
activationSet.add('relu')


class Tensor(list):
    def __init__(self, mat):
        # returns a Tensor which has a specific value
        # I actually want this to be a 2d array
        if not isinstance(mat, list) or not isinstance(mat[0], list):
            raise TypeError("Argument 1 must be a 2d array.")
        super(Tensor, self).__init__(mat)

    def __str__(self):
        s = ""
        for i in range(len(self)):
            s += str(self[i])
            if i != len(self)-1:
                s += "\n"
        return s


class Layer:
    def __init__(self, params: (int, int)):
        if not isinstance(params, tuple) or len(params) != 2 or not isinstance(params[0], int) or not isinstance(params[1], int):
            raise TypeError("Argument 1. 'params' must be a tuple(int,int)")
        self.params = params
        self.weight = []


def ReLU(target: Tensor) -> Tensor:
    # class for ReLU
    if not isinstance(target, Tensor):
        raise TypeError("Argument 1 must be of type 'Tensor'")
    # Get the weights of the layer

    for i in range(len(target)):
        for j in range(len(target[i])):
            if target[i][j] < 0:
                target[i][j] = 0
    return target


class Module:
    def __init__(self):
        # Initializes a new module where we have to keep track of our layers, activations, etc
        self.shape = []
        self.layers = []

    def add(self, target: Layer) -> None:
        # adds the target layer to the module

        if len(self.shape) == 0:
            self.shape.append(target.params[0])
        else:
            if self.shape[-1] != target.params[0]:
                raise TypeError("Target Layer Dim.0 of size " + str(
                    target.params[0]) + " not equal to model dim.-1 of size " + str(self.shape[-1]))
        self.shape.append(target.params[1])
        self.layers.append(target)

    def addAct(self, act):
        pass

    def __str__(self):
        return ("Module class @ " + str(id(self)))


test = Module()
test.add(Layer((6, 5)))
test.add(Layer((5, 6)))
test.addAct(ReLU())
print(test)
#test = Tensor([[-1, 1, -2], [-3, 4, -5], [-6, 7, -8]])
