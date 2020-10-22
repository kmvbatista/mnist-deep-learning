import pickle
import gzip

# Third-party libraries
import numpy as np

class Mnist_loader:

    def load_raw_data(self):
        f = gzip.open('./data/mnist.pkl.gz')
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        training_data, validation_data, test_data = u.load()
        f.close()
        return (training_data, validation_data, test_data)


    def load_data(self):
        """Return a tuple containing ``(training_data, validation_data,
        test_data)``. Based on ``load_data``, but the format is more
        convenient for use in our implementation of neural networks.

        In particular, ``training_data`` is a list containing 50,000
        2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
        containing the input image.  ``y`` is a 10-dimensional
        numpy.ndarray representing the unit vector corresponding to the
        correct digit for ``x``.

        ``validation_data`` and ``test_data`` are lists containing 10,000
        2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
        numpy.ndarry containing the input image, and ``y`` is the
        corresponding classification, i.e., the digit values (integers)
        corresponding to ``x``.

        Obviously, this means we're using slightly different formats for
        the training data and the validation / test data.  These formats
        turn out to be the most convenient for use in our neural network
        code."""
        raw_training_data, raw_validation_data, raw_test_data = self.load_raw_data()
        training_inputs = [np.reshape(x, (784, 1)) for x in raw_training_data[0]]
        training_results = [self.vectorized_result(y) for y in raw_training_data[1]]
        training_data = list(zip(training_inputs, training_results))
        validation_inputs = [np.reshape(x, (784, 1)) for x in raw_validation_data[0]]
        validation_data = list(zip(validation_inputs, raw_validation_data[1]))
        test_inputs = [np.reshape(x, (784, 1)) for x in raw_test_data[0]]
        test_data = list(zip(test_inputs, raw_test_data[1]))
        return (training_data, validation_data, test_data)


    def vectorized_result(self, j):
        """Return a 10-dimensional unit vector with a 1.0 in the jth
        position and zeroes elsewhere.  This is used to convert a digit
        (0...9) into a corresponding desired output from the neural
        network."""
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e
