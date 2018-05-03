import tensorflow as tf
import numpy as np
from pdrnn import general_logpmf, drnn_classification
from pdrnn import set_dilations_1, set_dilations_4, set_dilations_5, set_dilations_6
from pdrnn import pmf_entropy

class StochasticDilateNet:
    def __init__(self,
                 hidden_structs,
                 select_ranges,
                 n_layers,
                 n_classes,
                 n_evaluate,
                 optimizer,
                 dropout,
                 input_dims=1,
                 cell_type="RNN"):
        ''' Initialize the class
        
        Args:
                                 
        '''    

        # assign configurations
        self.hidden_structs = hidden_structs
        self.select_ranges = select_ranges
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.n_evaluate = n_evaluate
        self.input_dims = input_dims
        self.cell_type = cell_type
        
        
        # build the network
        self.inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
        char_embeddings = tf.get_variable("char_embeddings", [n_classes, input_dims])
        inputs_emb = tf.nn.embedding_lookup(char_embeddings, self.inputs)
        self.inputs_emb = tf.layers.dropout(inputs_emb, 0.3, training=dropout) 
        
        inputs_shape = tf.shape(self.inputs)
        self.n_steps = tf.cast(inputs_shape[0], tf.float32)
        
        self.labels = tf.placeholder(tf.int64, [None, None], name='labels')
        
        
        self.selections = [tf.placeholder(tf.int32, [], name='selections') for _ in xrange(n_layers)]
        self.dilations = []
        self.n_actions = []
        for l in xrange(n_layers):
            ranges = tf.range(select_ranges[l][0], select_ranges[l][1]+1)  #add 1 to achieve upper limit
            self.dilations.append(ranges[self.selections[l]]) 
            self.n_actions.append(select_ranges[l][1]-select_ranges[l][0]+1)
        
        #self.picks, self.struct_vars = set_dilations_4(self.n_actions,
        #                                               self.n_layers)
        
        self.picks, self.struct_vars = set_dilations_5(self.select_ranges,
                                                       self.n_actions,
                                                       self.n_layers)
        
        self.logits = drnn_classification(self.inputs_emb,
                                          self.hidden_structs,
                                          self.dilations,
                                          self.n_classes,
                                          self.n_evaluate,
                                          self.input_dims,
                                          self.cell_type)
                               
        # model weights
        self.weights = [v for v in tf.trainable_variables() if 'multi_dRNN_layer' in v.name]
        self.struct_param = [v for v in tf.trainable_variables() if 'struct_layer' in v.name]
        
        # entropy of pmf
        self.entropy = []
        for j in xrange(n_layers):
            self.entropy.append(pmf_entropy(self.struct_vars[j]))
                            
            
        # define modified loss for REINFORCE
        self.log_p_per_example \
        = sum([general_logpmf(dd, pp)
               for dd, pp in zip(self.selections, self.struct_vars)])
        
        
        # multiply a phantom term log_p to compute gradient over struct_param
        self.loss_per_example = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                labels=self.labels, logits=self.logits, name='cross_entropy_per_example')
        self.loss_for_w = tf.reduce_mean(self.loss_per_example)
        self.bpc_loss = self.loss_for_w / np.log(2.0)
        
        self.b = tf.get_variable('b', initializer=tf.constant(0.0), trainable=False)
        #b = tf.assign(b, (lambda_b*b+(1-lambda_b)*self.loss_for_w))
        self.loss_for_pi = tf.reduce_mean(self.log_p_per_example * (self.loss_per_example/np.log(2.0) - self.b))
        
        # model evaluation
        correct_pred = tf.equal(tf.argmax(self.logits,-1), self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        
        # define optimizers
        #optimizer = tf.train.RMSPropOptimizer(0.003)
        gvs = optimizer.compute_gradients(self.bpc_loss, var_list=self.weights)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        self.weights_train_op = optimizer.apply_gradients(capped_gvs)
                    
        struct_opt = tf.train.AdamOptimizer(learning_rate=0.005)    
        self.struct_train_ops = []
        for i in xrange(n_layers):
            self.struct_train_ops.append(
                     struct_opt.minimize(self.loss_for_pi,
                                         var_list=self.struct_param[i]))
                    
        
        # Add ops to save and restore variables.
        self.saver_for_all = tf.train.Saver(tf.trainable_variables(), max_to_keep=10)
        self.saver_for_w = tf.train.Saver(self.weights, max_to_keep=10)
        # save each layer separately
        self.savers = []
        for i in range(n_layers-1):
            vars_list = [v for v in tf.trainable_variables() if 'multi_dRNN_layer_{}'.format(i) in v.name]
            self.savers.append(tf.train.Saver(vars_list, name='saver_l_{}'.format(i)))
            
        # save final layer and post-processing layer together
        vars_list = [v for v in tf.trainable_variables() 
                     if 'multi_dRNN_layer_final' in v.name
                     or 'multi_dRNN_layer_{}'.format(n_layers-1) in v.name]
        self.savers.append(tf.train.Saver(vars_list, name='saver_l_{}'.format(n_layers-1)))
        
        
        
            
               