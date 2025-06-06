# network.py — From-Scratch Neural Network

This is a pure NumPy implementation of a feedforward neural network with support for backpropagation and basic training.

## Features

- Fully connected layers
- ReLU activation for hidden layers
- Softmax output layer
- Mean Squared Error (MSE) loss
- Forward pass and backward pass (backpropagation)
- Finite difference gradient descent (original)
- Real backprop implemented manually (current)

## Usage

### Define a Network

```python
from network import Network

net = Network([784, 64, 10])  # Input layer (28x28), one hidden layer, output layer (digits 0–9)
````

### Create Training Data

```python
from network import DataPoint
import numpy as np

# x = input (784,), y = one-hot encoded label (10,)
dp = DataPoint(inputs=x, expected=y)
```

### Train

```python
net.Learn([dp1, dp2, ...], eta=0.01)  # batch training
```

### Predict

```python
prediction = net.classify(input_array)  # returns the index of the highest softmax value
```

## Design

* Each layer is a `Layer` object with weights, biases, and gradients
* Forward pass stores activations and pre-activations (z-values)
* Backward pass calculates gradients using manual chain rule
* `applyGrads(eta)` updates parameters based on gradients

## Limitations

* No batching yet (training data is processed sample-by-sample)
* MSE used even for classification (can switch to cross-entropy)
* No saving/loading model weights (yet)

## Recommended Dataset

Tested with [MNIST](http://yann.lecun.com/exdb/mnist/), preprocessed using `mnist.pkl.gz`.

## File Structure

* `network.py`: the main neural network class
* `train.py`: example script to load MNIST, train the model

## License

MIT

