import tensorflow as tf
import numpy as np
from StochasticDropoutNet_unrolled import StochasticDropoutNet
from util import xor_data, extract_update_dict
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

model = StochasticDropoutNet(min_init_dropout_rate = 0.001, 
                             max_init_dropout_rate = 0.001,
                             train_batch_size = 50,
                             valid_batch_size = 100)
                             
inputs_train, targets_train = mnist.train.next_batch(55000)
inputs_train = inputs_train.reshape((-1, 28, 28, 1))
inputs_valid, targets_valid = mnist.validation.next_batch(5000)
inputs_valid = inputs_valid.reshape((-1, 28, 28, 1))

model.train_unroll(inputs_train, targets_train, inputs_valid, targets_valid, num_epochs = 400)
#model.train_fine_tune(inputs_train, targets_train, inputs_valid, targets_valid, num_epochs = 20)