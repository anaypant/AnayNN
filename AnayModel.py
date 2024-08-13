import numpy as np
import matplotlib.pyplot as plt
import pickle
# from keras.datasets import mnist
import math


class mlp:
    def __init__(self):
        # Initializes an AnayModel
        # We want the AnayModel to be able to add dense layers to create a network
        # We keep track of the network
        self.__shape = []

        # Keep track of weights
        self.__weights = []

        # Activations and set of all possible activations (using a hashset for efficiency)
        self.__activations = []
        self.__activationSet = set()
        self.__activationSet.add('sigmoid')
        self.__activationSet.add('relu')
        self.__activationSet.add('linear')
        self.__activationSet.add('softmax')
        self.__activationSet.add('tanh')
        # self.__activationSet.add('log_softmax')

        # Hidden/Output Neurons - This is saved for forward passes but is not initialized right now
        self.__hidden = None
        self.__outputs = None

        # Backwards pass - Losses, Adjustments
        self.__cost = None
        self.__errors = None
        self.__adjustments = None

    def __activate(self, x, layerId, deriv=False):
        # This will return the activation for a specific layer
        # layerId is the index for which layer we are adjusting
        # Important to figure out which activation we are performing

        # ------------------------- SIGMOID ------------------
        # Returns the sigmoid value of a layer x
        if self.__activations[layerId] == 'sigmoid':
            if deriv:
                return x * (1 - x)
            return 1 / (1 + np.exp(-x))
        # ------------------------- RELU ---------------------
        # Returns a relu value of a layer x
        elif self.__activations[layerId] == 'relu':
            if deriv:
                x[x < 0] = 0
                x[x > 0] = 1
                return x
            return np.maximum(0, x)
        # --------------------- LINEAR -----------------------
        elif self.__activations[layerId] == 'linear':
            if deriv:
                x[True] = 1
                return x
            return x
        # -------------------- TANH --------------------------
        elif self.__activations[layerId] == 'tanh':
            if deriv:
                return 1. - x**2  # This right now could result in nanning
            return np.tanh(x)
        # -------------------- SOFTMAX -----------------------
        elif self.__activations[layerId] == 'softmax':
            if deriv:
                return x * (1 - x)
            t = []
            for z in x:
                exps = np.exp(z - z.max())
                a = exps / np.sum(exps)
                t.append((a).tolist())
            t = np.reshape(t, (len(t), len(t[0])))
            return t
        # ----------------------------------------------------
        elif self.__activations[layerId] == 'log_softmax':
            if deriv:
                return x * (1 - x)
            t = []
            for z in x:
                c = z.max()
                logsumexp = np.log(np.exp(z-c).sum())
                t.append((z-c-logsumexp).tolist())
            t = np.reshape(t, (len(t), len(t[0])))
            return t
        else:
            # This should never be reached
            raise BaseException("Core: activation for layer " +
                                str(layerId) + "not recognized.")

    def addInputLayer(self, num_inputs):
        if len(self.__shape) != 0:
            raise BaseException("Input Layer already added.")
        # adds an input layer for number of input nodes
        self.__shape.append(num_inputs)
        # We don't want to add to the weights right now

    # ----------------- This version of addDenseLayer() takes in only the number of neurons in the previous layer
    def addDenseLayer(self, dims, activation):
        # adds a Dense layer to the network.
        if len(self.__shape) == 0:
            # This must be the input layer
            raise BaseException("Model does not contain an input layer.")
        if type(dims) != int:
            raise TypeError("'dims' must be a int (numNeurons)")

        if not self.__activationSet.__contains__(activation):
            raise TypeError("Activation " + activation +
                            " not found in possible activations")

        self.__shape.append(dims)

        # we want to now add to the weights
        n = len(self.__shape) - 1
        a = self.__shape[n-1]
        b = self.__shape[n]

        # re-initialize the weights properly
        lower, upper = -(1.0 / math.sqrt(a)), (1.0 / math.sqrt(b))
        self.__weights.append(2 * np.random.random((a,b))-1)

        # add the activation for this layer
        self.__activations.append(activation)

    # Fitting weights using forward passes and backwards passes

    def __forward_pass(self, x):
        # We want to essentially 'fit' the model and get predictions on our input
        # We can now reshape x to make sure it fits the correct shape
        # ------------ Initialize/reset hidden and output layers
        self.__outputs = []
        # ------------ Iterate through every layer and create hidden layers with
        if len(self.__shape) == 2:  # This means there are no hidden layers
            self.__outputs = self.__activate(np.dot(x, self.__weights[0]), 0)
        # For a multi-layer network, we will need hidden layers
        self.__hidden = []
        # ---------------------------------Multi Layer--------------------------
        for i in range(len(self.__shape) - 1):
            if i == 0:  # First layer
                self.__hidden.append(self.__activate(
                    np.dot(x, self.__weights[i]), i))
            elif i == len(self.__shape) - 2:
                self.__outputs = self.__activate(
                    np.dot(self.__hidden[i-1], self.__weights[i]), i)
            else:
                self.__hidden.append(self.__activate(
                    np.dot(self.__hidden[i-1], self.__weights[i]), i))

    def __backward_pass_with_crossentropy(self, x, y, learningRate=1, momentum=None):
        if self.__outputs is None:
            raise BaseException(
                "Core: Outputs not initiated before backwards pass")
        self.__errors = []
        self.__adjustments = []
        for i in range(len(self.__shape)-1, 0, -1):
            if i == len(self.__shape) - 1:
                self.__errors.insert(0, self.__outputs - y)
                self.__adjustments.insert(0, self.__errors[0] * learningRate * self.__activate(self.__outputs, -1, True))
                self.__cost = np.mean(np.abs(self.__errors))
            else:
                self.__errors.insert(0, self.__adjustments[0].dot(self.__weights[i].T))
                self.__adjustments.insert(0, self.__errors[0] * learningRate * self.__activate(self.__hidden[i-2], i, True))
            
        for i in range(len(self.__shape) - 1):
            if i == 0:
                self.__weights[i] -= np.dot(x.T, self.__adjustments[i])
            else:
                self.__weights[i] -= np.dot(self.__hidden[i-1].T, self.__adjustments[i])

    def __backward_pass(self, x, y, learningRate=1, momentum=None):
        # Get the losses for each layer
        # Using SGD to calculate the adjustments needed for each layer, and then implementing them
        # ---------------------- First y checks - make sure it has the right shape ----------------

        # Make sure our outputs are not none
        if self.__outputs is None:
            raise BaseException(
                "Core: Outputs not initiated before backwards pass")

        # Also make sure it is a 2d array
        # -----------------------------------Get loss of Network-----------------------------------
        # Reset all out params
        self.__errors = []
        self.__adjustments = []

        counter = len(self.__shape) - 2  # This is for weights, activations
        for i in range(len(self.__shape) - 1):
            if i == 0:
                # THIS SHOULD ALWAYS WORK
                self.__cost = y-self.__outputs
                self.__errors.append(self.__cost)
                self.__cost = abs(self.__cost)
                self.__cost = np.mean(self.__cost)
                self.__adjustments.append(
                    self.__errors[i] * learningRate * self.__activate(self.__outputs, counter, deriv=True))

            else:
                self.__errors.append(
                    self.__adjustments[i-1].dot(self.__weights[counter+1].T))
                self.__adjustments.append(
                    self.__errors[i] * learningRate * self.__activate(self.__hidden[counter], counter, deriv=True))

            if i == len(self.__shape) - 2:
                self.__weights[counter] += np.dot(x.T, self.__adjustments[i])
            else:
                self.__weights[counter] += np.dot(
                    self.__hidden[counter-1].T, self.__adjustments[i])

            counter -= 1

    def fit(self, x, y, iterations:int, trialX=None, trialY=None , learningRate=1, pBar=True, plot=False, validation_clip=1.0, batch_fitting=True):
        # -------------------- Make sure that x matches the shape of our weights
        # X has to be a 2d array
        if not type(x) == list and type(x[0]) == list:
            raise TypeError("x is not a 2d array")
        if len(x[0]) != self.__shape[0]:
            raise BaseException(
                "Dimensions of samples in x do not match size " + str(self.__shape[0]))
        if not type(y) == list and type(y[0]) == list:
            raise TypeError("Core: y is not a 2d array")
        if len(y[0]) != self.__shape[-1]:
            raise BaseException(
                "Dimensions of samples in y do not match size " + str(self.__shape[-1]))
        if len(y) != len(x):
            raise TypeError("x samples length " + str(len(x)) +
                            " does not match y samples length " + str(len(y)))

        if trialX is not None and trialY is not None:
            if not type(trialX) == list and type(trialX[0]) == list:
                raise TypeError("Validation X is not a 2d array")
            if len(trialX[0]) != self.__shape[0]:
                raise BaseException(
                    "Dimensions of samples in trialX do not match size " + str(self.__shape[0]))
            if not type(trialY) == list and type(trialY[0] == list):
                raise TypeError("Validation Y is not a 2d array")
            if len(trialY) != len(trialX):
                raise TypeError("Validation X length " + str(len(trialX)) +
                                " does not match Validation Y length " + str(len(trialY)))
            if len(trialY[0]) != self.__shape[-1]:
                raise BaseException(
                    "Dimensions of samples in trialY do not match size " + str(self.__shape[-1]))
            tX = np.reshape(trialX, (len(trialX), self.__shape[0]))
            tY = np.reshape(trialY, (len(trialY), self.__shape[-1]))



        x = np.reshape(x, (len(x), self.__shape[0]))
        y = np.reshape(y, (len(y), self.__shape[-1]))
        # right now idc about iterations lets do one pass
        if pBar:
            num_blocks = 20
            if iterations < num_blocks:
                num_blocks = iterations
            # meaning every 50 iterations a block will be added
            iters_per_block = iterations//num_blocks
            # number of blocks filled will be total iters // i//self.iters_per_block
            blockMsg = ""
            for z in range(num_blocks):
                blockMsg += " "
            current_block_index = 0

        pX, pY, pvY = None, None, None
        if plot:
            pX = []
            pY = []
            pvY = []

        for iteration in range(iterations):

            if not batch_fitting:
                pass

            self.__forward_pass(x)
            #self.__backward_pass(x, y, learningRate=learningRate)
            self.__backward_pass_with_crossentropy(
                x, y, learningRate=learningRate)
            # print(self.__adjustments)
            # print(self.__weights)
            # figure out the validation scores

            if trialX is not None and trialY is not None:

                if self.__shape[-1] != 1:
                    validation_score = self.__validate(tX, tY)
                else:
                    validation_score = self.__validate(tX, tY, regression=True)
            else:
                validation_score = "N/A"

            if pBar:
                if (iteration+1)//iters_per_block == current_block_index+1:
                    blockMsg = blockMsg[:current_block_index] + \
                        "=" + blockMsg[current_block_index+1:]
                    current_block_index += 1
                # this tells us how many blocks are filled
                # Add the cost and the value of it
                rD = 5
                rDV = 4
                costRounded = str(round(self.__cost, rD))
                if len(costRounded) < rD+2:
                    for z in range(rD+2 - len(costRounded)):
                        costRounded += " "
                
                if type(validation_score) != str:
                    vScoreRounded = str(round(validation_score, rDV))
                    if len(vScoreRounded) < rDV+2:
                        for z in range(rDV+2-len(vScoreRounded)):
                            vScoreRounded += " "
                else:
                    vScoreRounded = validation_score

                print("Progress: " + "|"+blockMsg+"|    Cost: " + costRounded +
                      " Val: " + vScoreRounded + " i="+str(iteration+1), end="\r")
            if plot:
                pX.append(iteration)
                pY.append(self.__cost)
                pvY.append(validation_score)

            if type(validation_score) != str:
                if validation_score > validation_clip:
                    # return out of the function
                    print("Training terminated to exceeded clip score of " +
                        str(validation_clip) + "at i=" + str(iteration))
                    return

        if pBar:
            print()
        if plot:
            plt.plot(pX, pY)
            if type(validation_score) != str: plt.plot(pX, pvY)
            plt.ylim(-1.0, 1.0)
            plt.show()

    def description(self):
        # print out a neat description of the model
        # we want to display each layer with its activation
        finalMsg = "\n"
        finalMsg += "-----------AnayModel-----------\n"
        # For each layer, we want to print out the number of neurons in the layer

        for layer in range(len(self.__shape)):
            # This gives us each layer iteration
            msg = "L"+str(layer) + "  | " + "Size: " + str(self.__shape[layer])
            if layer != 0:
                msg += "  |   Activation: " + self.__activations[layer-1]
            else:
                msg += "  <-- Input Layer"
            finalMsg += msg + "\n"
            finalMsg += "-------------------------------\n"
        return finalMsg

    def __get_prediction(self, inputs, validated=False):
        if not validated:
            if len(inputs) != self.__shape[0]:
                print(
                    "Core: inputs for prediction not accurate to model size #get_prediction")
                return
            inputs = np.reshape(inputs, (1, self.__shape[0]))
        hidden = []
        outputs = []
        if len(self.__weights) == 1:
            outputs = self.__activate(
                np.dot(inputs, self.__weights[0]), 0)
        else:
            for i in range(len(self.__weights)):
                if i == 0:
                    hidden.append(self.__activate(
                        np.dot(inputs, self.__weights[0]), i))
                elif i == len(self.__weights)-1:
                    outputs = self.__activate(
                        np.dot(hidden[i-1], self.__weights[i]), i)
                else:
                    hidden.append(self.__activate(
                        np.dot(hidden[i-1], self.__weights[i]), i))
        return outputs

    def predict(self, inputs, rounded=False):
        # input must be an array
        if not isinstance(inputs, list) and not isinstance(inputs[0], list):
            raise TypeError("Input for predition is not a type 2d array.")
        if len(inputs[0]) != self.__shape[0]:
            raise TypeError("Input length " + str(len(inputs)) +
                            " for prediction not accurate to model size " + str(self.__shape[0]))
        inputs = np.reshape(inputs, (len(inputs), self.__shape[0]))

        hidden = []
        outputs = []
        if len(self.__weights) == 1:
            outputs = self.__activate(
                np.dot(inputs, self.__weights[0]), 0)
        else:
            for i in range(len(self.__weights)):
                if i == 0:
                    hidden.append(self.__activate(
                        np.dot(inputs, self.__weights[0]), i))
                elif i == len(self.__weights)-1:
                    outputs = self.__activate(
                        np.dot(hidden[i-1], self.__weights[i]), i)
                else:
                    hidden.append(self.__activate(
                        np.dot(hidden[i-1], self.__weights[i]), i))
        if rounded:
            for z in range(len(outputs)):
                # find index of max value in results
                index_of_max_value = np.argmax(outputs[z])
                for k in range(len(outputs[z])):
                    if k != index_of_max_value:
                        outputs[z][k] = 0
                    else:
                        outputs[z][k] = 1
        return outputs

    def __validate(self, x, y, regression=False):
        # ---------------------- Goal is to take these x's and y's and make a validation score of the inputs we are not trying
        if len(x[0]) != self.__shape[0]:
            print("Core: inputs for prediction not accurate to model size #validate")
            return
        if len(x) != len(y):
            print("Core: input len " + str(len(x)) +
                  " does not match y len " + str(len(y)) + " #validate")
        num_trials = len(x)
        # inputs = np.reshape(x, (len(x), self.__shape[0]))
        # figure out the number of inputs

        results = (self.__get_prediction(x, validated=True))
        # pick the best one

        num_correct = 0
        for z in range(len(results)):
            # find index of max value in results
            if not regression:
                index_of_max_value = np.argmax(results[z])
                for k in range(len(results[z])):
                    if k != index_of_max_value:
                        results[z][k] = 0
                    else:
                        results[z][k] = 1
                num_correct += 1
                for w in range(len(results[z])):
                    if results[z][w] != y[z][w]:
                        num_correct -= 1
                        break
            elif regression and np.round(results[z]).astype('uint8') == y[z]:
                num_correct += 1
        return round(float(num_correct/num_trials), 3)

    def get_activations(self):
        # Returns a list of all possible activations
        s = self.__activationSet.copy()
        return s

    def __open_model(self):
        # makes the model open so that it can be read from
        return {'w': self.__weights, 'shape': self.__shape, 'acts': self.__activations}

    def save_model(self, filename):
        # making sure that the filename ends in .bin
        l = len(filename)-4
        if l < 0:
            raise TypeError("Filename must end in .bin")
        if not filename[l:len(filename)] == ".bin":
            raise TypeError("Filename must end in .bin")
        # saves the model using pickle to a specific file name
        with open(filename, "wb") as f:
            pickle.dump(self.__open_model(), f)

    def load_model(self, filename):
        # loads an AnayModel with the specified filename
        with open(filename, "rb") as f:
            d = pickle.load(f)
        if not isinstance(d, dict):
            raise BaseException("File did not load successfully")
        # cast A to everything in this fricking object that we have already created
        self.__weights = d['w']
        self.__activations = d['acts']
        self.__shape = d['shape']


