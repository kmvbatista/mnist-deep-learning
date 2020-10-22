import numpy


weights_file= 'weights.npy'
biases_file= 'biases.npy'

def save_training_result(weights, biases):
  numpy.save(weights_file, weights)
  numpy.save(biases_file, weights)
  

def load_trained_result():
  weights = numpy.load(weights_file)
  biases = numpy.load(biases_file)
  return (weights, biases)