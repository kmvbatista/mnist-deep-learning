import pickle
import gzip

# Third-party libraries
import numpy as np


def load_raw_data():
    f = gzip.open('./data/mnist.pkl.gz')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    training_data, validation_data, test_data = u.load()
    f.close()
    return (training_data, validation_data, test_data)


def load_data():
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
    raw_tr_data, raw_val_data, raw_test_data = load_raw_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in raw_tr_data[0]]
    training_results = [vectorized_result(y) for y in raw_tr_data[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
