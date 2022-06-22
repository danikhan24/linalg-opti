import numpy as np
import scipy as sc


def accuracy(y_pred, y_actual, threshold=0.5):
    return (threshold - np.linalg.norm(y_pred - y_actual) > 0) * 1


# Base class
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # computes the output Y of a layer for a given input X
    def forward_propagation(self, input):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError


class AdamOptim:
    def __init__(self, eta=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m_dw = 0
        self.v_dw = 0
        self.m_db = 0
        self.v_db = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta

    def update(self, eta, t, w, b, dw, db):
        t = 1
        ## dw, db are from current minibatch
        ## momentum beta 1
        # *** weights *** #
        self.m_dw = self.beta1 * self.m_dw + (1 - self.beta1) * dw
        # *** biases *** #
        self.m_db = self.beta1 * self.m_db + (1 - self.beta1) * db

        ## rms beta 2
        # *** weights *** #
        self.v_dw = self.beta2 * self.v_dw + (1 - self.beta2) * (dw ** 2)
        # *** biases *** #
        self.v_db = self.beta2 * self.v_db + (1 - self.beta2) * (db ** 2)

        ## bias correction
        m_dw_corr = self.m_dw / (1 - self.beta1 ** t)
        m_db_corr = self.m_db / (1 - self.beta1 ** t)
        v_dw_corr = self.v_dw / (1 - self.beta2 ** t)
        v_db_corr = self.v_db / (1 - self.beta2 ** t)

        ## update weights and biases
        w = w - self.eta * (m_dw_corr / (np.sqrt(v_dw_corr) + self.epsilon))
        b = b - self.eta * (m_db_corr / (np.sqrt(v_db_corr) + self.epsilon))
        return w, b


# Inherited
class DenseLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.weights_old = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        adam = AdamOptim()
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        self.weights_old = self.weights

        self.weights, self.bias = adam.update(
            t=1,
            eta=learning_rate,
            w=self.weights,
            b=self.bias,
            dw=weights_error,
            db=output_error,
        )

        # update parameters
        return input_error


# Layer of Activation functions


class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    # returns the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error


#


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.accuracy_metric = []
        self.error_metric = []

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input

    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []
        err = 0
        acc = 0

        # run network over all samples
        for i in range(samples):

            # forward propagation
            output = input_data[i]

            for layer in self.layers:
                output = layer.forward_propagation(output)

            err += self.loss(input_data[i], output)
            acc += accuracy(input_data[i], output)
            result.append(output)

        err /= samples
        acc = acc / samples
        print(err)
        print(acc)
        print("Prediction error=%f   accuracy=%f " % (err, acc))

        return result, err, acc

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        print(" Fitting Neural Network")
        # sample dimension first
        samples = len(x_train)
        acc_array = []
        err_array = []

        # training loop
        for i in range(epochs):
            err = 0
            acc = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss and accuracy
                err += self.loss(y_train[j], output)
                acc += accuracy(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error and accuracy on all samples
            err /= samples
            acc = acc / samples
            # yeah 2 lines above are completely different operations

            err_array.append(err)
            acc_array.append(acc)
            print("epoch %d/%d   error=%f   accuracy=%f " % (i + 1, epochs, err, acc))

        self.accuracy_metric = acc_array
        self.error_metric = err_array
