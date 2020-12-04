import numpy as np
from math import floor


class FFNN:
    """
    Feed Foward Neural network. Takes a list of nodes in a given layer
    and a list of activation function for each hidden layer.
    Can do both classification and regression depending on the
    choice of activation in the last hidden layer.

    This function takes a flexible no. of nodes and layers.

    Params:
    --------
        layers: list
            list of no of nodes for each layer, including
            input and output layer.

        activation_functions: list of strings
            list of strings specifying each activation function
            for each hidden layer.

    Returns
    --------
        y: vector
            Vector of predicted outputs. If back_prop function is called
            before prediction, then we train the network.
    """
    def __init__(self, layers, activation_functions):
        self.layers = layers
        self.weights = []
        self.biases = []
        self.activation_functions = []
        self.activation_derivatives = []

        if len(activation_functions)+1 != len(layers):
            print("Warning: No. of activation functions does not match number of hidden layers")


        for i in range(len(layers) - 1):
            # He initialization and normal distribution of weights
            self.weights.append(
                np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2 / layers[i])
            )

            self.biases.append(np.zeros((1, layers[i + 1])))

            if activation_functions[i] == "sigmoid":
                self.activation_functions.append(lambda x : 1 / (1 + np.exp(-x)))
                self.activation_derivatives.append(lambda a : a * (1 - a))

            elif activation_functions[i] == "relu":
                self.activation_functions.append(lambda x : np.maximum(x, 0))
                self.activation_derivatives.append(lambda a : np.where(a > 0, 1, 0))

            elif activation_functions[i] == "relu6":
                self.activation_functions.append(lambda x : np.minimum(np.maximum(x, 0), 6))
                self.activation_derivatives.append(lambda a : np.ones(a.shape) * np.logical_and(a > 0, a < 6))

            elif activation_functions[i] == "leaky_relu":
                self.activation_functions.append(lambda x : np.where(x > 0, x, x * 0.01))
                self.activation_derivatives.append(lambda a : np.where(a > 0, 1, 0.01))

            elif activation_functions[i] == "tanh":
                self.activation_functions.append(lambda x : np.tanh(x))
                self.activation_derivatives.append(lambda a : 1 - a**2)

            elif activation_functions[i] == "softmax":
                self.activation_functions.append(lambda x : np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1))
                self.activation_derivatives.append(lambda a : a*(1-a))

            elif activation_functions[i] == "binary":
                activation_functions[i].append(lambda x : np.where(x >= 0.5, 1, 0))
                activation_derivatives[i].append(lambda a : 0)

            elif activation_functions[i] == "identity":
                self.activation_functions.append(lambda x: x)
                self.activation_derivatives.append(lambda a: np.ones(a.shape))

            else:
                print("Warning: Activation function", activation_functions[i], "for layer", i, "is not implemented.")


    def forward_pass(self, x):
        """
        Predicts a output based on initial input and
        pass it through network and activates with activation Functions
        for each hidden layer.

        Params:
            x: Array

        Returns:
            Array of activated output from input and hidden layers
        """
        for i in range(len(self.layers) - 1):
            x = self.activation_functions[i]((x @ self.weights[i]) + self.biases[i])
        return x

    def back_prop(self, x, y, learning_rate, epochs, mini_batches):
        """
        This function trains the neural network
        for classification or regression.

        Params:
        --------
            x: array
                array of shape (no. of observations, no. of features).
            y: array
                vector of output data.
            learning_rate: float
                the learning rate of the stochastic gradient descent.
            epochs: int
                specifies the no. of times the whole dataset has
                been passed through and back the network.
            mini_batches: int
                number of mini-batches to part the dataset in.

        Returns:
        --------
            None
        """
        N = x.shape[0]
        batch_size = floor(N/mini_batches)
        for e in range(epochs):
            for batch in range(batch_size):
                random_idx = np.random.randint(0, N, size=batch_size)
                x_batch = x[random_idx]
                y_batch = y[random_idx]
                z = []
                a = [x_batch]
                L = len(self.layers)

                # Forward pass, store a list and z list
                for i in range(L - 1):
                    z.append((a[i] @ self.weights[i]) + self.biases[i])
                    a.append(self.activation_functions[i](z[i]))

                # Initialize for last layer, [-1]
                gradient_cost = a[-1] - y_batch  # True for MSE and Cross Entropy (given softmax activation for last layer)
                delta = (
                    self.activation_derivatives[-1](a[-1]) * gradient_cost
                )  # Local gradient, delta_L (last layer)
                self.dB = [np.mean(delta, axis=0)]
                self.dW = [a[-2].T @ delta]

                # Back propagate, fill these lists: dW and dB
                for i in range(2, L):
                    """
                    Loop going between all hidden layers,
                    and goes from last layer to first,
                    must start from 2 because we use -i.
                    """
                    delta = (
                        delta @ self.weights[-i + 1].T
                    ) * self.activation_derivatives[-i](a[-i])
                    self.dB.append(np.mean(delta, axis=0))
                    self.dW.append(a[-i - 1].T @ delta)

                # Loop over layers and update all weights and biases with lists dW and dB
                for i in range(L - 1):
                    self.weights[i] -= learning_rate * self.dW[-i - 1] / N
                    self.biases[i] -= learning_rate * self.dB[-i - 1]
