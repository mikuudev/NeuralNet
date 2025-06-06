import numpy as np


class DataPoint:
    def __init__(self, inputs, expected):
        self.inputs = np.array(inputs)
        self.expected = np.array(expected)


class Layer:
    def __init__(self, numNodesIn: int, numNodesOut: int):
        self.numNodesIn = numNodesIn
        self.numNodesOut = numNodesOut

        # initialize weights + biases with random values (gaussian noise for the win)
        self.weights = np.random.randn(self.numNodesOut, self.numNodesIn)
        self.biases = np.random.randn(self.numNodesOut)

        # prepare empty gradients to accumulate during training
        self.costGradientW = np.zeros((self.numNodesOut, self.numNodesIn))
        self.costGradientB = np.zeros(self.numNodesOut)

    def applyGrads(self, eta):
        # actually update weights + biases using the stored gradients and learning rate
        self.weights -= self.costGradientW * eta
        self.biases -= self.costGradientB * eta

    def calcOutputs(self, inputs):
        z = np.dot(self.weights, inputs) + self.biases
        return self.ReLU(z)

    def ReLU(self, x):
        return np.maximum(0, x)

    @staticmethod
    def ReLUDerivative(x):
        return (x > 0).astype(float)  # derivative is 1 where x > 0, else 0


class Network:
    def __init__(self, layerSizes):
        self.layers = [Layer(layerSizes[i], layerSizes[i + 1]) for i in range(len(layerSizes) - 1)]
        self.activations = []        # post-activation outputs (ReLU/softmax)
        self.weightedInputs = []     # pre-activation z values

    def calcOutputs(self, inputs):
        self.activations = [np.array(inputs)]  # first activation = input
        self.weightedInputs = []

        for i, layer in enumerate(self.layers):
            z = np.dot(layer.weights, inputs) + layer.biases
            self.weightedInputs.append(z)

            # softmax for output layer, ReLU otherwise
            inputs = self.softmax(z) if i == len(self.layers) - 1 else layer.ReLU(z)
            self.activations.append(inputs)

        return inputs

    def classify(self, inputs):
        outputs = self.calcOutputs(inputs)
        return int(np.argmax(outputs))

    def singleCost(self, output, expected):
        return np.sum((output - expected) ** 2)

    def singleCostDerivative(self, output, expected):
        return 2 * (output - expected)

    def calcCost(self, data):
        outputs = self.calcOutputs(data.inputs)
        diff = outputs - data.expected
        return np.sum(diff**2) / len(outputs)

    def avgCost(self, dataset):
        totalCost = sum(self.calcCost(data) for data in dataset)
        return totalCost / len(dataset)

    def Learn(self, trainingBatch: list[DataPoint], eta: float):
        # update gradients for all points in batch
        for datapoint in trainingBatch:
            self.updateAllGradients(datapoint)

        # apply the gradients after batch is processed
        self.ApplyAllGradients(eta)

        # clear gradients to prepare for next batch
        self.ClearAllGradients()

    def ApplyAllGradients(self, eta):
        for layer in self.layers:
            layer.applyGrads(eta)

    def ClearAllGradients(self):
        for layer in self.layers:
            layer.costGradientW.fill(0)
            layer.costGradientB.fill(0)

    def softmax(self, x):
        exps = np.exp(x - np.max(x))  # fix overflow (numerical stability hack)
        return exps / np.sum(exps)

    def calcOutputNodeValues(self, expected):
        output = self.activations[-1]
        z = self.weightedInputs[-1]

        # we use MSE, not cross-entropy, so softmax derivative is ignored
        cost_deriv = self.singleCostDerivative(output, expected)
        activation_deriv = np.ones_like(z)  # don't touch it, softmax + MSE = lazy combo

        return cost_deriv * activation_deriv

    def updateAllGradients(self, data: DataPoint):
        self.calcOutputs(data.inputs)
        nodeValues = self.calcOutputNodeValues(data.expected)

        # output layer gets gradients directly from cost derivative
        output_layer = self.layers[-1]
        last_activation = self.activations[-2]
        output_layer.costGradientW += np.outer(nodeValues, last_activation)
        output_layer.costGradientB += nodeValues

        # walk backwards through the rest of the layers (classic backprop fashion)
        for layer_index in range(len(self.layers) - 2, -1, -1):
            current_layer = self.layers[layer_index]
            next_layer = self.layers[layer_index + 1]

            # compute new node values (δ) for this layer using next layer's info
            nodeValues = self.calcHiddenValues(layer_index, next_layer, nodeValues)

            prev_activation = self.activations[layer_index]
            current_layer.costGradientW += np.outer(nodeValues, prev_activation)
            current_layer.costGradientB += nodeValues

    def calcHiddenValues(self, this_index, nextLayer: Layer, nextNodeValues):
        thisLayer = self.layers[this_index]
        newNodeValues = np.zeros(thisLayer.numNodesOut)

        # go node-by-node and compute δ for each using the next layer
        for i in range(thisLayer.numNodesOut):
            for j in range(nextLayer.numNodesOut):
                newNodeValues[i] += nextLayer.weights[j, i] * nextNodeValues[j]

            # apply ReLU derivative to kill off dead neurons (rip)
            newNodeValues[i] *= Layer.ReLUDerivative(self.weightedInputs[this_index][i])

        return newNodeValues
