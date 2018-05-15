import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import tensorflow as tf
from models.classification_models import drnn_classification

data_path = "./penn_tree_clean.npz"
n_steps = 100 #length of input sequence
input_dims = 128 # char embedding dimension
n_classes = 49 # vocab size

# model config
cell_type = "GRU"
assert(cell_type in ["RNN", "LSTM", "GRU"])
hidden_structs = [256] * 7
dilations = [1,2,4,6,8,32,64]
assert(len(hidden_structs) == len(dilations))

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
        
        return batch_x.T, batch_y
    
    def get_validation_batches(self):
        """
        A generator for all validation data in mini batches.
        """        
        for batch_id in range(self.n_batches):
            offset_base = np.arange(batch_id*self.val_batch_size, 
                                    (batch_id+1)*self.val_batch_size)
            selections = np.tile(np.arange(self.n_steps)[:,np.newaxis],
                                 self.val_batch_size) + offset_base
            batch_x = self.data_dict['valid'][selections]
            batch_y = self.data_dict['valid'][offset_base+self.n_steps]
            yield batch_x.T, batch_y
        
ptb_data1 = PTB_data(np.load(data_path), 100, 16384)


# build computation graph
tf.reset_default_graph()
x = tf.placeholder(tf.int32, [None, n_steps])
y = tf.placeholder(tf.int32, [None,])
dropout = tf.placeholder(tf.bool, [])

char_embeddings = tf.get_variable("char_embeddings", [n_classes, input_dims])
x_emb = tf.nn.embedding_lookup(char_embeddings, x)
x_emb = tf.layers.dropout(x_emb, 0.3, training=dropout)

global_step = tf.Variable(0, name='global_step', trainable=False)

# build prediction graph
print "==> Building a dRNN with %s cells" %cell_type
pred = drnn_classification(x_emb, hidden_structs, dilations, n_steps, n_classes, input_dims, cell_type)

# build loss and optimizer
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=y))
bpc_cost = cost / np.log(2.0)
optimizer = tf.train.AdamOptimizer(learning_rate)
gvs = optimizer.compute_gradients(bpc_cost)
capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)


# define session
sess_config = tf.ConfigProto(allow_soft_placement=True,
                             log_device_placement=False,
                             intra_op_parallelism_threads=20,
                             inter_op_parallelism_threads=4)
sess_config.gpu_options.allow_growth = True


sess = tf.Session(config=sess_config)
# initialize all the parameters
sess.run(tf.global_variables_initializer())


for step in xrange(10000000):
    batch_x, batch_y = ptb_data.random_train_batch(batch_size, n_steps)    

    feed_dict = {
        x : batch_x,
        y : batch_y,
        dropout: True
    }
    bpc_cost_, _ = sess.run([bpc_cost, train_op], feed_dict=feed_dict)    
    
    if (step + 1) % 10 == 0:
        print "Iter " + str(step + 1) + ", Minibatch Loss: " + "{:.6f}".format(bpc_cost_)
             
    if (step + 1) % 300 == 0:
        
        # validation performance
        batch_bpcs = []
        for batch_x, batch_y in ptb_data.get_validation_batches():
            feed_dict = {
                x : batch_x,
                y : batch_y,
                dropout: False
            }
            bpc_cost_ = sess.run([bpc_cost], feed_dict=feed_dict) 
            batch_bpcs.append(bpc_cost_)
        validation_bpc = np.mean(batch_bpcs)
        print "========> Validation BPC: " + "{:.6f}".format(validation_bpc) + " over %d batches" % len(batch_bpcs)
