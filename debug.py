import tensorflow as tf
import numpy as np
from pdrnn import triangular_pmf, build_pdrnn, compute_loss
from StochasticDilateNet import StochasticDilateNet
from scipy.stats import bernoulli
import hdf5storage as hdst

data = hdst.loadmat('/mnt/hdd1/kqian3/rl_struct/seq')
seq = data['seq']

tf.reset_default_graph()
hidden_structs = [8]*2
init_params = [(1.0,0.5),(1.0,0.5)]
lambda_b = 0.0
model = StochasticDilateNet(hidden_structs,
                            init_params,
                            n_layers=2,
                            n_classes=2,
                            input_dims=1,
                            cell_type="RNN")
# define session
sess_config = tf.ConfigProto(allow_soft_placement=True,
                             log_device_placement=False,
                             intra_op_parallelism_threads=20,
                             inter_op_parallelism_threads=4)
sess_config.gpu_options.allow_growth = True

loss = []
for rate in range(1,101):
    with tf.Session(config=sess_config) as sess:
        # initialize all the parameters
        sess.run(tf.global_variables_initializer())
        
        for step in range(500):
            feed_dict = {model.inputs: seq,
                         model.labels: seq[0,:,0],
                         model.rates[-1]: rate}
            sess.run(model.weights_train_op, feed_dict=feed_dict)
                    
        loss_value = sess.run(model.loss_for_w, feed_dict=feed_dict)
        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
        loss.append(loss_value)
        print 'Rate = {}, loss = {}'.format(rate, loss_value)    