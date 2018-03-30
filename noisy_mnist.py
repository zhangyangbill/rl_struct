import os
import json
import argparse

import tensorflow as tf
import numpy as np
import hdf5storage as hdst
from StochasticDilateNet import StochasticDilateNet
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

from pdrnn import list_enqueue, list_enqueue_batch, \
                  find_closest_element_batch
from pdrnn import make_noisy_mnist   

parser = argparse.ArgumentParser()  
parser.add_argument('config', type=str,
                    help='path to configuration json file')
args = parser.parse_args() 
config_filename = args.config

config = json.load(open(config_filename))

logdir = config['logdir']
gpu_index = config['gpu_index']
support = config['support']
pad_to = config['pad_to']


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_index



n_layers = len(support)
hidden_structs = [20]*n_layers
lambda_b = 0.9
model = StochasticDilateNet(hidden_structs,
                            support,
                            n_layers=n_layers,
                            n_classes=10,
                            n_evaluate=1,
                            input_dims=1,
                            cell_type="RNN")
# define session
sess_config = tf.ConfigProto(allow_soft_placement=True,
                             log_device_placement=False,
                             intra_op_parallelism_threads=20,
                             inter_op_parallelism_threads=4)
sess_config.gpu_options.allow_growth = True



def restore_layers(savers, ks, logdir, bottom=0):
    for i in xrange(bottom, n_layers):
        path = "{}model-{}".format(logdir, 
                                   history_steps[ks[i]])
        savers[i].restore(sess, path)  


sess = tf.Session(config=sess_config)
# initialize all the parameters
sess.run(tf.global_variables_initializer())

bottom = 0
num_w_op = 10
history_rates = [[] for _ in xrange(n_layers)]
history_steps = []
feed_dict = {}

masks = [np.full((1,support[i][1]),-np.inf) for i in xrange(n_layers)]

for step in xrange(1000000):
    X, Y = mnist.train.next_batch(64)
    inputs_train, target_train = make_noisy_mnist(X, Y, pad_to)    
    
    if step % num_w_op == 0:
        if step != 0:
            #X, Y = mnist.validation.next_batch(64)
            #inputs_valid, target_valid = make_noisy_mnist(X, Y, 2000)
            #feed_dict[model.inputs] = inputs_valid
            #feed_dict[model.labels] = target_valid
            
            loss_ = sess.run(model.loss_for_w, feed_dict=feed_dict)
            if step == num_w_op:
                sess.run(tf.assign(model.b, loss_))
            else:
                sess.run(tf.assign(model.b, (lambda_b*model.b+(1-lambda_b)*loss_)))
                
            model.saver_for_w.save(sess, 
                                   "{}model".format(logdir), 
                                   global_step=step)
            list_enqueue_batch(rates, history_rates)
            list_enqueue(step, history_steps)
                        
            
        if step >= 2*num_w_op:
            sess.run(model.struct_train_ops, feed_dict=feed_dict)    
            
            
        picks = sess.run(model.picks, feed_dict=feed_dict)
        rates = sess.run(model.dilations, feed_dict={s: p for s, p in zip(model.selections, picks)})
        if len(history_steps) >= 1:
            ks = find_closest_element_batch(rates, history_rates)
            print 'Step {}, Draw = {},\n history_rates = {},\n Replace = {}'.format(step, rates, history_rates, ks)
            restore_layers(model.savers, ks, logdir, bottom)
            
        
    for d, r in zip(model.selections, picks):
        feed_dict[d] = r   
    feed_dict[model.inputs] = inputs_train
    feed_dict[model.labels] = target_train
        
    sess.run(model.weights_train_op, feed_dict=feed_dict)
    
    if step % num_w_op == num_w_op-1:
        #feed_dict[model.inputs] = inputs_valid
        #feed_dict[model.labels] = target_valid
        struct_vars, entropy, b, loss_value, accuracy = \
        sess.run([model.struct_vars, model.entropy,
                  model.b, model.loss_for_w, model.accuracy],
                 feed_dict=feed_dict)
        print('\n Step {}, Rates {}, b = {}, loss = {}, accuracy = {}\n Entropy = {}\n'.format(
            step, rates, b, loss_value, accuracy, entropy))
        # check entropy
        for l in xrange(n_layers):
            if entropy[l] < 1.5 and not np.isnan(entropy[l]):
                model.struct_train_ops[l] = tf.no_op('Stop')
                masks[l][0,np.argmax(struct_vars[l])] = 0.0
                feed_dict[model.struct_vars[l]] = masks[l]
        # save logits        
        if step % 100 == 99:
            for k in xrange(bottom, n_layers):
                hdst.savemat('{}pmf_{}_{}'.format(logdir,step,k),
                             {'pmf':struct_vars[k][0]})
        if loss_value < 1e-6:
            break