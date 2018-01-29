import os
import argparse 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

parser = argparse.ArgumentParser()  
parser.add_argument("LOG_DIR", help="LOG_DIR")
args = parser.parse_args() 


import tensorflow as tf
import numpy as np
from StochasticDropoutNet_unrolled import StochasticDropoutNet
from util import xor_data, extract_update_dict
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

model = StochasticDropoutNet(min_init_dropout_rate = 0.4, 
                             max_init_dropout_rate = 0.6,
                             train_batch_size = 50,
                             valid_batch_size = 500,
                             unroll_steps = 5,
                             num_weight_train_steps = 4,
                             head = 401,
                             log_dir = args.LOG_DIR)

#states = tf.train.get_checkpoint_state('/mnt/hdd1/kqian3/rl_struct')
#checkpoint_paths = states.all_model_checkpoint_paths
model.saver.restore(model.sess, "{}model_unroll-400".format(args.LOG_DIR))
                             
inputs_train, targets_train = mnist.train.next_batch(55000)
inputs_train = inputs_train.reshape((-1, 28, 28, 1))
inputs_valid, targets_valid = mnist.validation.next_batch(5000)
inputs_valid = inputs_valid.reshape((-1, 28, 28, 1))

model.train_unroll(inputs_train, targets_train, inputs_valid, targets_valid, num_epochs = 701)
#model.train_fine_tune(inputs_train, targets_train, inputs_valid, targets_valid, num_epochs = 300)