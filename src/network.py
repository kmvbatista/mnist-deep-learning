import random
import numpy
import numpy as np
import file_handler

class Network(object):

    def __init__(self, qtt_neurons_each_layer, training_data):
        """The list ``qtt_neurons_each_layer`` contains the number of neurons in the
          respective layers of the network.  For example, if the list
          was [2, 3, 1] then it would be a three-layer network, with the
          first layer containing 2 neurons, the second layer 3 neurons,
          and the third layer 1 neuron.  The biases and weights for the
          network are initialized randomly, using a Gaussian
          distribution with mean 0, and variance 1.  Note that the first
          layer is assumed to be an input layer, and by convention we
          won't set any biases for those neurons, since biases are only
          ever used in computing the outputs from later layers."""
        self.training_data = training_data
        self.num_layers = len(qtt_neurons_each_layer)
        self.qtt_neurons_each_layer = qtt_neurons_each_layer
        self.biases = [np.random.randn(y, 1)
                       for y in qtt_neurons_each_layer[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(qtt_neurons_each_layer[:-1], qtt_neurons_each_layer[1:])]

    def train(self, epochs, mini_batch_size, learning_rate, test_data):
        self.train_stochastically(epochs, mini_batch_size, learning_rate,
                                                 test_data=None)
        self.save_training_result()

    def train_and_test(self, epochs, mini_batch_size, learning_rate, test_data):
        self.train_stochastically(epochs, mini_batch_size, learning_rate,
                                                 test_data)
        self.save_training_result()

    def save_training_result(self):
        file_handler.save_training_result(weights=self.weights, biases=self.biases)

    def train_stochastically(self, epochs, mini_batch_size, learning_rate,
                                     test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data:
            n_test = len(test_data)
        training_data_quantity = len(self.training_data)
        for j in range(epochs):
            random.shuffle(self.training_data)
            mini_batches = self.get_mini_batches(mini_batch_size, training_data_quantity)
            for mini_batch in mini_batches:
                self.update_weights_by_mini_batch(mini_batch, learning_rate)
            if test_data:
                print('Epoch {0}: {1} / {2}'.format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_weights_by_mini_batch(self, mini_batch, learning_rate):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)````
        ."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for activation, expected_result in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backpropagate(activation, expected_result)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(learning_rate/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(learning_rate/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backpropagate(self, activation, expected_result):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activations = [activation] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.get_cost_derivative(activations[-1], expected_result) * self.get_sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = self.get_nabla_w(delta, activations[-2])
        for l in range(2, self.num_layers):
            z = zs[-l]
            z_sigmoid_prime = self.get_sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * z_sigmoid_prime
            nabla_b[-l] = delta
            nabla_w[-l] = self.get_nabla_w(delta, activations[-l-1])
        return (nabla_b, nabla_w)

    def get_nabla_w(self, delta, layer_activations):
        return np.dot(delta, layer_activations.transpose())

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        z=0
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a)+b
            a = self.sigmoid(z)
        return 

    def get_mini_batches(self, mini_batch_size, training_data_quantity):
        return [
                self.training_data[k:k+mini_batch_size]
                for k in range(0, training_data_quantity, mini_batch_size)]

    def get_cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives 
        partial C_x /partial a
        for the output activations."""
        return (output_activations-y)

    def sigmoid(self, z):
        return 1.0/(1.0+numpy.exp(-z))

    def get_sigmoid_prime(self, z):
        """Derivative of the sigmoid function."""
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
            
