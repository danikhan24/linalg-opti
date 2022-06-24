import numpy as np


def accuracy(y_pred, y_actual, threshold=0.5):
    return (np.linalg.norm(y_pred - y_actual) < threshold) * 1


# Base class
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # Forward propogation
    def forward(self, input):
        raise NotImplementedError

    # Backward propogation
    def backward(self, output_error, learning_rate):
        raise NotImplementedError


class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, perturbation=1e-11):
        self.mdw = 0
        self.vdw = 0
        self.mdb = 0
        self.vdb = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.perturbation = perturbation
        self.eta = learning_rate

    def update_params(self, learning_rate, w, b, dw, db):

        ## Momentum beta_1
        self.mdw = self.beta1 * self.mdw + (1 - self.beta1) * dw
        self.mdb = self.beta1 * self.mdb + (1 - self.beta1) * db
        # RMS beta_2
        self.vdw = self.beta2 * self.vdw + (1 - self.beta2) * (dw**2)
        self.vdb = self.beta2 * self.vdb + (1 - self.beta2) * (db**2)
        # A i,k
        A_mdw = self.mdw / (1 - self.beta1)
        A_mdb = self.mdb / (1 - self.beta1)
        A_vdw = self.vdw / (1 - self.beta2)
        A_vdb = self.vdb / (1 - self.beta2)

        ## Updating weights, biases; really small perturbation to ensure that we don't divide by zero
        w = w - learning_rate * (A_mdw / (np.sqrt(A_vdw) + self.perturbation))
        b = b - learning_rate * (A_mdb / (np.sqrt(A_vdb) + self.perturbation))
        return w, b


# Inherited
class DenseLayer(Layer):
    # input_size = number of input_neurons
    # output_size = number of output_neurons
    def __init__(self, input_neurons, output_neurons):
        self.weights = np.random.rand(input_neurons, output_neurons) - 0.5
        self.weights_old = np.random.rand(input_neurons, output_neurons) - 0.5
        self.bias = np.random.rand(1, output_neurons) - 0.5

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_error, learning_rate):
        adam = Adam()
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        self.weights_old = self.weights

        # Use adam to update weights and biases
        self.weights, self.bias = adam.update_params(
            learning_rate=learning_rate,
            w=self.weights,
            b=self.bias,
            dw=weights_error,
            db=output_error,
        )

        return input_error


# Layer of Activation functions


class ActivationLayer(Layer):
    def __init__(self, activation, grad_activation):
        self.activation = activation
        self.grad_activation = grad_activation

    # Just activation
    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_error, learning_rate):
        return self.grad_activation(self.input) * output_error


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.grad_loss = None
        self.accuracy_metric = []
        self.error_metric = []
        self.testing_accuracy = []
        self.testing_error = []

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, grad_loss):
        self.loss = loss
        self.grad_loss = grad_loss

    def forward_propagation(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = np.array([input_data[i]])

            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)

        return result

    # predict output for given input

    def predict(self, input_data, y_test):
        # sample dimension first
        samples = len(input_data)
        result = []
        err = 0
        acc = 0

        # run network over all samples
        for i in range(samples):

            # forward propagation
            output = np.array([input_data[i]])

            for layer in self.layers:
                output = layer.forward(output)

            err += self.loss(np.array([y_test[i]]), output)
            acc += accuracy(np.array([y_test[i]]), output)
            result.append(output)

        err /= samples
        acc = acc / samples
        print("Prediction error=%f   accuracy=%f \n" % (err, acc))

        return result, err, acc

    # train the network
    def fit(self, x_train, y_train, x_test, y_test, epochs, learning_rate):
        print("Fitting Neural Network")
        # sample dimension first
        samples = len(x_train)
        samples_test = len(x_test)
        acc_array = []
        err_array = []
        test_acc_array = []
        test_err_array = []

        # training loop
        for i in range(1, epochs+1):
            err = 0
            acc = 0
            test_acc = 0
            test_err = 0
            for j in range(samples):
                # forward propagation
                output = np.array([x_train[j]])
                for layer in self.layers:
                    output = layer.forward(output)

                # compute loss and accuracy
                err += self.loss(np.array([y_train[j]]), output)
                acc += accuracy(np.array([y_train[j]]), output)

                # backward propagation
                error = self.grad_loss(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)

            for j in range(samples_test):
                output_test = np.array([x_test[j]])
                for layer in self.layers:
                    output_test = layer.forward(output_test)
                test_err += self.loss(np.array([y_test[j]]), output_test)
                test_acc += accuracy(np.array([y_test[j]]), output_test)

            # calculate average error and accuracy on all samples
            err /= samples
            acc = acc / samples
            test_err /= samples_test
            test_acc = test_acc / samples_test
            # yeah 2 lines above are completely different operations

            err_array.append(err)
            acc_array.append(acc)
            test_err_array.append(test_err)
            test_acc_array.append(test_acc)
            if i % 100 == 0:
                print(
                "epoch %d/%d   error=%f   accuracy=%f   test_err=%f   test_acc=%f"
                % (i, epochs, err, acc, test_err, test_acc)
                )

        self.accuracy_metric = acc_array
        self.error_metric = err_array
        self.testing_accuracy = test_acc_array
        self.testing_error = test_err
