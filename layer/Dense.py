import numpy as np


class Dense: #Linear Dense Layer of Neurons to perform matrix multiplication and dot product
    def __init__(self, params:tuple[int, int]):
        #number of input neurons, number of output neurons
        self.__weights = 2 * np.random.random((params[0], params[1])) - 1

    