import tensorflow as tf
import numpy as np
from pdrnn import triangular_pmf, build_pdrnn


class StochasticDilateNet:
    def __init__(self,
                 hidden_structs,
                 init_params,
                 n_layers,
                 n_classes,
                 input_dims=1,
                 cell_type="RNN"):
        ''' Initialize the class
        
        Args:
                                 
        '''    

        # assign configurations
        self.hidden_structs = hidden_structs
        self.init_params = init_params
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.input_dims = input_dims
        self.cell_type = cell_type
        
        
        # build the network
        self.inputs = tf.placeholder(tf.float32, shape = [None, None, input_dims], name = 'inputs')
        inputs_shape = tf.shape(self.inputs)
        self.n_steps = tf.cast(inputs_shape[0], tf.float32)
        self.labels = tf.placeholder(tf.int64, [None, None])
        
        self.logits, self.struct_vars, self.rates = build_pdrnn(self.inputs,
                                                                self.hidden_structs,
                                                                self.init_params,
                                                                self.n_layers,
                                                                self.n_steps,
                                                                self.n_classes,
                                                                self.input_dims,
                                                                self.cell_type)
                               
        # model weights
        self.weights = [v for v in tf.trainable_variables() if 'multi_dRNN_layer' in v.name]
        struct_param = [v for v in tf.trainable_variables() if 'struct_layer' in v.name]
        self.struct_param = struct_param[-2:] # select which params to update
            
            
        # define modified loss for REINFORCE
        self.log_p_per_example \
        = sum([tf.log(triangular_pmf(dd, pp[0], pp[1], self.n_steps)) 
               for dd, pp in zip(self.rates, self.struct_vars)])
        
        
        # multiply a phantom term log_p to compute gradient over struct_param
        self.loss_per_example = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                labels=self.labels, logits=self.logits, name='cross_entropy_per_example')
        self.loss_for_w = tf.reduce_mean(self.loss_per_example)
        
        b = tf.get_variable('b', initializer=tf.constant(0.0))
        #b = tf.assign(b, (lambda_b*b+(1-lambda_b)*self.loss_for_w))
        self.b = b
        self.loss_for_pi = tf.reduce_mean(self.log_p_per_example * (self.loss_per_example - b))
        
        # model evaluation
        correct_pred = tf.equal(tf.argmax(self.logits,-1), self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        
        # define optimizers
        self.weights_train_op = tf.train.AdamOptimizer()\
                                .minimize(self.loss_for_w,
                                          var_list = self.weights)
        
        self.struct_train_op = tf.train.AdagradOptimizer(learning_rate=10)\
                                .minimize(self.loss_for_pi,
                                          var_list = self.struct_param)
            
        # clip mu and sigma
        self.struct_clip_op = []
        for mu, sigma in self.struct_vars:
            mu_clipped = tf.clip_by_value(mu, 0.5, self.n_steps-0.5)
            sigma_clipped = tf.clip_by_value(sigma, 1.0, self.n_steps)
            structs_clipped = (tf.assign(mu,mu_clipped),
                               tf.assign(sigma,sigma_clipped))
            self.struct_clip_op.append(structs_clipped)
                    
        
        # Add ops to save and restore all the variables.
        self.saver_for_w = tf.train.Saver(self.weights, max_to_keep=20)
            
               