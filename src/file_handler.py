import json
from distutils import file_util
import os
from pip._internal.utils import filesystem

weights_file= 'weights.json'
biases_file= 'biases.json'

def save_training_result(weights, biases):
  with open(weights_file, 'w') as outfile:
    json.dump(weights, outfile)
  with open(biases_file, 'w') as outfile:
    json.dump(biases, outfile)
  # file_util.write_file(weights_file, weights)
  # file_util.write_file(biases_file, biases)

def load_trained_result():
  weights =[]
  biases =[]
  with open(weights_file) as inFile:
    weights=json.load(inFile)
  with open(biases_file) as inFile:
    biases=json.load(inFile)