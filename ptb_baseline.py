import os
import json
import argparse

import tensorflow as tf
import numpy as np
from StochasticDilateNet_ptb import StochasticDilateNet
import cPickle as pickle


parser = argparse.ArgumentParser()  
parser.add_argument('config', type=str,
                    help='path to configuration json file')
args = parser.parse_args() 
config_filename = args.config

config = json.load(open(config_filename))

logdir = config['logdir']
gpu_index = config['gpu_index']
support = config['support']
seq_len = config['seq_len']
valid_batch_size = config['valid_batch_size']
emb_dim = config['emb_dim']
optimizer = getattr(tf.train, config['optimizer'])
lr = config['learning_rate']
dropout = config['dropout']


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_index

class PTB_data(object):
    def __init__(self, data_dict, n_steps, val_batch_size):
        self.data_dict = data_dict
        train_size = self.data_dict['train'].shape[0]
        self.train_rng = np.arange(train_size-n_steps-1)
        validation_size = self.data_dict['valid'].shape[0]
        self.n_batches = (validation_size-n_steps-1) / val_batch_size
        self.val_batch_size = val_batch_size
        self.n_steps = n_steps
        
    def random_train_batch(self, batch_size, n_steps):
        """
        Randomly sample a batch of sequences of length n_steps from training set.
        """
        offsets = np.random.choice(self.train_rng, batch_size)
        selections = np.tile(np.arange(n_steps)[:,np.newaxis],
                             batch_size) + offsets
        
        batch_x = self.data_dict['train'][selections]
        batch_y = self.data_dict['train'][offsets+n_steps]
        
        return batch_x, batch_y
    
    def get_validation_batches(self):
        """
        A generator for all validation data in mini batches.
        """        
        for batch_id in range(self.n_batches):
            offset_base = np.arange(batch_id*self.val_batch_size, 
                                    (batch_id+1)*self.val_batch_size)
            selections = np.tile(np.arange(self.n_steps)[:,np.newaxis],
                                 self.val_batch_size) + offset_base
            batch_x = self.data_dict['test'][selections]
            batch_y = self.data_dict['test'][offset_base+self.n_steps]
            yield batch_x, batch_y

data_path = "./penn_tree_clean.npz"             
ptb_data = PTB_data(np.load(data_path), seq_len, valid_batch_size)

n_layers = len(support)
hidden_structs = [256]*n_layers
lambda_b = 0.9
model = StochasticDilateNet(hidden_structs,
                            support,
                            n_layers=n_layers,
                            n_classes=49,
                            n_evaluate=1,
                            optimizer=optimizer(lr),
                            input_dims=emb_dim,
                            cell_type="GRU")
# define session
sess_config = tf.ConfigProto(allow_soft_placement=True,
                             log_device_placement=False,
                             intra_op_parallelism_threads=20,
                             inter_op_parallelism_threads=4)
sess_config.gpu_options.allow_growth = True


sess = tf.Session(config=sess_config)
# initialize all the parameters
sess.run(tf.global_variables_initializer())



states = tf.train.get_checkpoint_state(logdir)
if states is not None:
    checkpoint_paths = states.all_model_checkpoint_paths
    model.saver_for_w.recover_last_checkpoints(checkpoint_paths)
    model.saver_for_w.restore(sess, tf.train.latest_checkpoint(logdir))


feed_dict = {d: r for d, r in zip(model.dilations, [1,2,4,8,16,32,64])}  

batch_size = 64
losses = []


for step in range(100000):
    inputs_train, target_train = ptb_data.random_train_batch(64, 100)  
    feed_dict[model.inputs] = inputs_train
    feed_dict[model.labels] = target_train
    #feed_dict[model.dropout] = dropout
        
    _, loss_value = sess.run([model.weights_train_op, model.bpc_loss], 
                              feed_dict=feed_dict)
    if step == 0:
        b = loss_value
    else:
        b = 0.9*b + 0.1*loss_value
    if np.abs(loss_value-b) > 0.6:
        hs, gvs, capped_gvs = sess.run([model.test_out, model.gvs, model.capped_gvs], 
                                       feed_dict=feed_dict)
        probe_dict = {'inputs':inputs_train, 
                      'target':target_train,
                      'hs':hs,
                      'gvs':gvs,
                      'capped_gvs':capped_gvs}
        with open('{}probe_{}.p'.format(logdir, step), "wb") as fp:
            pickle.dump(probe_dict, fp)
                
    if step % 10 == 0:
        print('Step {}, loss = {}'.format(step, loss_value))

    if step % 300 == 0 and step != 0:
        # validation performance
        batch_bpcs = []
        for inputs_val, target_val in ptb_data.get_validation_batches():
            feed_dict[model.inputs] = inputs_val
            feed_dict[model.labels] = target_val
            #feed_dict[model.dropout] = False
            bpc_cost_ = sess.run(model.bpc_loss, feed_dict=feed_dict) 
            batch_bpcs.append(bpc_cost_)
        validation_bpc = np.mean(batch_bpcs)
        print "========> Validation BPC: " + "{:.6f}".format(validation_bpc)  
        model.saver_for_w.save(sess, "{}model".format(logdir), global_step=step)