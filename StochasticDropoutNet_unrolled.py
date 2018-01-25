import tensorflow as tf
import numpy as np
import random
from util import LBFGS, graph_replace, extract_update_dict
from optimizers import Adagrad, Adam
from tensorflow.examples.tutorials.mnist import input_data

class StochasticDropoutNet:
    def __init__(self,
                 train_batch_size = 32,
                 valid_batch_size = 32,
                 num_weight_train_steps = 4,
                 weight_delta_max_count = 128,
                 min_dropout_rate = 0.0001,
                 max_dropout_rate = 0.9999,
                 min_init_dropout_rate = 0.2,
                 max_init_dropout_rate = 0.8,
                 unroll_steps = 3):
        ''' Initialize the class
        
        Args:
        train_batch_size - batch size for training
        valid_batch_size - batch size for test
        input_dtype - input data type
        input_size - a tuple input shape
        target_dtype - target data type
        target_size - a tuple of target shape
        num_weight_train_steps - number of gradient steps to train the weights 
                                 before the structural parameters are updated
        weight_delta_max_count - maximum number of weight deltas stored
        min_dropout_rate - the minimum dropout probability allowed
        max_dropout_rate - the maximum dropout probability allowed
        min_init_dropout_rate - the minimum initial dropout probability allowed
        max_init_dropout_rate - the maximum initial dropout probability allowed
        unroll_steps - the number of unrolled steps
        '''
        
        
        # arrays recording network weights and structural parameters
        self.weights = []
        self.struct_param = []
        self.keeps = [] # stores instances of whether each node should be kept
        
        # assign configurations
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.num_weight_train_steps = num_weight_train_steps
        
        
        # define dropout rate limits
        self.min_dropout_rate = min_dropout_rate
        self.max_dropout_rate = max_dropout_rate
        self.min_init_dropout_rate = min_init_dropout_rate
        self.max_init_dropout_rate = max_init_dropout_rate
        
        
        # build the network
        self.build_graph()
        
        # define modified loss for REINFORCE
        self.log_p_per_example \
        = sum([tf.reduce_sum(tf.log(pp) * (1 - kk) + tf.log(1 - pp) * kk,
                             axis = [1,2,3]) 
               for pp, kk in zip(self.struct_param, self.keeps)])
        # multiply a phantom term log_p to compute gradient over struct_param
        self.loss = tf.reduce_mean(self.loss_per_example * 
                                   (tf.stop_gradient(1 - self.log_p_per_example)
                                    + self.log_p_per_example))
        
        
        # define unrolling parameters
        self.unroll_steps = unroll_steps
        self.inputs_adapt = [tf.placeholder_with_default(shape = self.inputs.shape,
                                                         input = self.inputs)
                             for i in xrange(self.unroll_steps)]
        self.targets_adapt = [tf.placeholder_with_default(shape = self.targets.shape,
                                                          input = self.targets)
                             for i in xrange(self.unroll_steps)]
        
        # define unrolling optimizers
        self.global_epoch = tf.Variable(0, name='global_epoch', trainable=False)
        #boundaries = [200, 300]
        #values = [0.0001, 0.00005, 0.00002]
        #learning_rate = tf.train.piecewise_constant(self.global_epoch, boundaries, values)
        self.increment_global_epoch_op = tf.assign(self.global_epoch, self.global_epoch+1)
        w_opt = Adagrad()
        updates = w_opt.get_updates(self.loss, self.weights)
        self.weights_train_op = tf.group(*updates, name="weights_train_op")
        
        update_dict = extract_update_dict(updates)
        
        
        cur_update_dict = graph_replace(update_dict, 
                                        {self.inputs: self.inputs_adapt[0],
                                         self.targets: self.targets_adapt[0]})
        
        for i in xrange(self.unroll_steps-1):
            # Change the inputs
            update_dict_adapt = graph_replace(update_dict, 
                                              {self.inputs: self.inputs_adapt[i+1],
                                               self.targets: self.targets_adapt[i+1]})
            # Compute variable updates given the previous iteration's updated variable
            cur_update_dict = graph_replace(update_dict_adapt, cur_update_dict)
            
        
        # Final unrolled loss uses the parameters at the last time step
        self.unrolled_loss = graph_replace(self.loss, cur_update_dict)
        self.struct_train_op = tf.train.AdamOptimizer(learning_rate = 0.001)\
                                .minimize(self.unrolled_loss,
                                          var_list = self.struct_param)
        # clip the value
        self.struct_clip_op = []
        for var in self.struct_param:
            var_clipped = tf.clip_by_value(var, self.min_dropout_rate, self.max_dropout_rate)
            self.struct_clip_op.append(tf.assign(var, var_clipped))
        
        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver(max_to_keep=50)
        
        # define session
        self.sess = tf.Session()
        
        # initialize all the parameters
        self.sess.run(tf.global_variables_initializer())
        
        # optimizer counters
        self.weight_counter = 0
        self.train_step_counter = 0
        self.struct_counter = 0
        self.epoch_counter = 0
        
        # define tensorboard operations
        tf.summary.scalar('losses', self.loss)
        self.merged_summary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('./log/',
                                                  graph = self.sess.graph)
        self.train_losses = []
        self.valid_losses = []
        
        
    def conv2d(self, inputs, 
               filter_width,
               filter_height,
               num_filters,
               var_scope,
               strides = [1, 1, 1, 1],
               padding = 'VALID',
               act_fun = None,
               dropout = True,
               residual = True,
               bias = False):
        '''Perform 2D convolution with stochastic dropout
        
        Args:
        input - a tensor of shape [batch_size, width, height, num_in_filters]
        num_filters - number of output filters
        var_scope - variable scope, a string
        strides - A list of ints. 1-D tensor of length 4.
                  The stride of the sliding window for each dimension of input
        padding - A string from: "SAME", "VALID". The type of padding algorithm to use.
        act_fun - the activation function to be applied. Must be directly callable functions.
                  'None' means linear activation function.
        dropout - True if stochastic dropout is applied
        residual - True if ResNet is applied
        
        Returns:
        output - a tensor of shape [batch_size, width, height, num_filters]
        '''
        
        # get the shapes
        _, height, width, num_in_filters = inputs.get_shape().as_list()
        
        # generate weight parameters
        with tf.variable_scope(var_scope):
            w_shape = [filter_height, filter_width, num_in_filters, num_filters]
            w = tf.get_variable('weights', w_shape)
            self.weights += [w]
          
            
        # perform 2D convolution
        _conv_output = tf.nn.conv2d(inputs,
                                    w,
                                    strides,
                                    padding)
        
        # obtain the output size
        out_shape = tf.shape(_conv_output)
        out_shape_list = _conv_output.get_shape().as_list()
        
        # apply bias
        if bias:
            with tf.variable_scope(var_scope):
                b = tf.get_variable('biases', initializer=tf.constant(0.01, shape=out_shape_list[1:]), trainable=True)
                self.weights += [b]
            _conv_output = _conv_output + b
        
        # apply activation function
        if act_fun is not None:
            _act_output = act_fun(_conv_output)
        else:
            _act_output = _conv_output
        
        # apply dropout
        if dropout:
            # generate dropout rate parameters
            with tf.variable_scope(var_scope):
                pi = tf.get_variable('drop_prob', 
                                     shape = out_shape_list[1:],
                                     initializer = tf.random_uniform_initializer(minval = self.min_init_dropout_rate,
                                                                                 maxval = self.max_init_dropout_rate))
                self.struct_param += [pi]

            # sample from Bernoulli distribution or from default
            keep = tf.placeholder_with_default(
                  tf.where(tf.random_uniform(out_shape) > pi,
                           tf.ones(out_shape),
                           tf.zeros(out_shape)),
                  shape = out_shape_list)
            # stop the gradient to prevent NaN over pi
            keep = tf.stop_gradient(keep)
            self.keeps += [keep]
            
            # generate the final output
            output = tf.multiply(_act_output, keep)
            
        else:
            output = _act_output
            
        
        # apply residual
        if residual:
            output = output + inputs
            
        return output

    
                    
    def train_unroll(self,
                     train_inputs,
                     train_targets,
                     valid_inputs,
                     valid_targets,
                     num_epochs = 10):
        '''
        This function performs the unrolled training process.
        
        Args:
        train_input - a numpy array of training input data, 1st dim is token
        train_targets - a numpy array of training targets, 1st dim is token
        valid_inputs - a numpy array of validation input data, 1st dim is token
        valid_targets - a numpy array of training targets, 1st dim is token
        num_epochs - number of epochs
        '''
        
        # define dataset
        mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
        inputs_test, targets_test = mnist.test.next_batch(10000)
        inputs_test = inputs_test.reshape((-1, 28, 28, 1))
                
        # determine number of tokens
        num_tokens_train = train_inputs.shape[0]
        num_tokens_valid = valid_inputs.shape[0]
        num_train_batches = num_tokens_train / self.train_batch_size
        array_num_tokens_valid = range(num_tokens_valid)
        
        # iterate epochs
        epoch_id = 0
        batch_id = 0
        struct_step_id = 0
        # shuffle the data
        shuffle = np.random.permutation(num_tokens_train)
        shuffle_next = np.random.permutation(num_tokens_train)
        while epoch_id < num_epochs:
            ###### perform weights training
            for w_id in xrange(self.num_weight_train_steps):
                selected_tokens_train = shuffle[batch_id * self.train_batch_size :
                                                (batch_id+1) * self.train_batch_size]
                feed_dict = {self.inputs: train_inputs[selected_tokens_train,...],
                             self.targets: train_targets[selected_tokens_train,...]}
                self.sess.run(self.weights_train_op,
                              feed_dict = feed_dict)
                
                # output training information
                loss, accuracy = self.sess.run([self.loss,
                                                self.accuracy],
                                               feed_dict = feed_dict)
                print('Epoch {}, weight train batch {}, step {}, loss = {}, accuracy = {}'.format(epoch_id,
                                                                                                  batch_id,
                                                                                                  w_id,
                                                                                                  loss,
                                                                                                  accuracy))
                # print struct param
                if batch_id % 100 == 0:
                    #struct_param_value = self.sess.run(self.struct_param, feed_dict = feed_dict)
                    #print('struct_param_value = {}'.format(struct_param_value))
                    print('test accuracy %g' % self.sess.run(self.accuracy, feed_dict={self.inputs: inputs_test,
                                                                                      self.targets: targets_test}))
                    
                # increment the counters
                batch_id = batch_id + 1
                if batch_id >= num_train_batches:
                    # reshuffle the data
                    shuffle = shuffle_next
                    shuffle_next = np.random.permutation(num_tokens_train)
                    batch_id = 0
                    struct_step_id = 0
                    epoch_id += 1
                    # Save the variables to disk.
                    save_path = self.saver.save(self.sess, "/mnt/hdd1/kqian3/rl_struct/model_unroll", global_step=epoch_id)
                    print("Model saved in file: %s" % save_path)
                    
                    
                    
                if epoch_id >= num_epochs:
                    break
                    
            ###### perform struct training
            selected_tokens_valid = random.sample(array_num_tokens_valid, 
                                                  self.valid_batch_size)
            # locate the data for unroll weight training steps
            # special handling at the boundary between epochs
            selected_tokens_train = [shuffle[b * self.train_batch_size :
                                             (b+1) * self.train_batch_size]
                                     for b in xrange(batch_id, 
                                                     min(batch_id+self.unroll_steps,
                                                         num_train_batches))] + \
                                    [shuffle_next[b * self.train_batch_size :
                                             (b+1) * self.train_batch_size]
                                     for b in xrange(self.unroll_steps 
                                                     - min(batch_id+self.unroll_steps,
                                                           num_train_batches)
                                                     + batch_id)]
            feed_dict = dict(zip([self.inputs,
                                  self.targets] +
                                 self.inputs_adapt +
                                 self.targets_adapt,
                                 [valid_inputs[selected_tokens_valid,...],
                                  valid_targets[selected_tokens_valid,...]] +
                                 list(train_inputs[selected_tokens_train,...]) +
                                 list(train_targets[selected_tokens_train,...])))
            #self.sess.run(self.struct_train_op,
            #              feed_dict = feed_dict)
            
            #if struct_step_id % 100 == 0:
            #    struct_param_value = self.sess.run(self.struct_param, feed_dict = feed_dict)
            #    print('struct_param_value = {}'.format(struct_param_value))
            
            #bias_value = self.sess.run([self.weights[-1], self.weights[-3]], feed_dict = feed_dict)
            #print('bias_value = {}'.format(bias_value))
            
            #self.sess.run(self.struct_clip_op,
            #              feed_dict = feed_dict)
            
            
            
            # increment the counters
            struct_step_id += 1
            
            # ouput training information
            loss, accuracy, struct_param = self.sess.run([self.loss, 
                                                          self.accuracy,
                                                          self.struct_param],
                                                         feed_dict = feed_dict)
            print('Epoch {}, struct train step {}, loss = {}, accuracy = {}'.format(epoch_id,
                                                                                    struct_step_id,
                                                                                    loss,
                                                                                    accuracy))
            #print(struct_param)
            self.struct_param_value = struct_param
                            
    
    def train_fine_tune(self,
                        train_inputs,
                        train_targets,
                        valid_inputs,
                        valid_targets,
                        num_epochs = 10):
        '''
        This function trains the weights with quantized keep probabilities.
        
        Args:
        train_input - a numpy array of training input data, 1st dim is token
        train_targets - a numpy array of training targets, 1st dim is token
        valid_inputs - a numpy array of validation input data, 1st dim is token
        valid_targets - a numpy array of training targets, 1st dim is token
        num_epochs - number of epochs
        '''
        
        # determine number of tokens
        num_tokens_train = train_inputs.shape[0]
        num_tokens_valid = valid_inputs.shape[0]
        num_train_batches = num_tokens_train / self.train_batch_size
        array_num_tokens_valid = range(num_tokens_valid)
        
        # run the keep probability
        feed_dict = {self.inputs: train_inputs[0:1,...],
                     self.targets: train_targets[0:1,...]}
        self.keep_value = self.sess.run(self.keeps,
                                        feed_dict = feed_dict)
        
        # iterate epochs
        epoch_id = 0
        batch_id = 0
        # shuffle the data
        shuffle = np.random.permutation(num_tokens_train)
        shuffle_next = np.random.permutation(num_tokens_train)
        while epoch_id < num_epochs:
            ###### perform weights training
            for w_id in xrange(self.num_weight_train_steps):
                selected_tokens_train = shuffle[batch_id * self.train_batch_size :
                                                (batch_id+1) * self.train_batch_size]
                feed_dict = dict(zip([self.inputs,
                                      self.targets]
                                     + self.keeps,
                                     [train_inputs[selected_tokens_train,...],
                                     train_targets[selected_tokens_train,...]]
                                     + self.keep_value))
                self.sess.run(self.weights_train_op,
                              feed_dict = feed_dict)
                
                # output training information
                loss, accuracy = self.sess.run([self.loss,
                                                self.accuracy],
                                               feed_dict = feed_dict)
                print('Epoch {}, weight train batch {}, step {}, loss = {}, accuracy = {}'.format(epoch_id,
                                                                                                  batch_id,
                                                                                                  w_id,
                                                                                                  loss,
                                                                                                  accuracy))
   
                # increment the counters
                batch_id = batch_id + 1
                if batch_id >= num_train_batches:
                    # reshuffle the data
                    shuffle = shuffle_next
                    shuffle_next = np.random.permutation(num_tokens_train)
                    batch_id = 0
                    struct_step_id = 0
                    epoch_id += 1
                    save_path = self.saver.save(self.sess, "/mnt/hdd1/kqian3/rl_struct/model_fine", global_step=epoch_id)
                    print("Model saved in file: %s" % save_path)
                    
                if epoch_id >= num_epochs:
                    break
                    
            ###### perform struct training
            selected_tokens_valid = random.sample(array_num_tokens_valid, 
                                                  self.valid_batch_size)
            feed_dict = dict(zip([self.inputs,
                                  self.targets] +
                                 self.keeps,
                                 [valid_inputs[selected_tokens_valid,...],
                                  valid_targets[selected_tokens_valid,...]] +
                                 self.keep_value))
            
            
            # ouput training information
            loss, accuracy = self.sess.run([self.loss, 
                                            self.accuracy],
                                           feed_dict = feed_dict)
            print('Validation loss = {}, accuracy = {}'.format(loss,
                                                               accuracy))
    
    def build_graph(self):
        ''' This function builds the dropout net architecture'''
    
        self.inputs = tf.placeholder(tf.float32, shape = [None, 28, 28, 1], name = 'inputs')
        self.targets = tf.placeholder(tf.int32, shape = [None,], name = 'targets')
        
        hidden = self.conv2d(self.inputs, 5, 5, 32, 
                             var_scope = 'conv0',
                             residual = False,
                             padding = 'SAME',
                             act_fun = tf.nn.softplus,
                             bias = True)
        #hidden = self.conv2d(hidden, 5, 5, 128, 
        #                     var_scope = 'conv1',
        #                     residual = True,
        #                     padding = 'SAME',
        #                     act_fun = tf.nn.softplus,
        #                     bias = False)
        hidden = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')
        
        hidden = self.conv2d(hidden, 5, 5, 64, 
                             var_scope = 'conv2',
                             residual = False,
                             padding = 'SAME',
                             act_fun = tf.nn.softplus,
                             bias = True)

        #hidden = self.conv2d(hidden, 5, 5, 128, 
        #                     var_scope = 'conv3',
        #                     residual = True,
        #                     padding = 'SAME',
        #                     act_fun = tf.nn.softplus,
        #                     bias = False)
        hidden = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')

        
        hidden = self.conv2d(hidden, 7, 7, 1024,
                             var_scope = 'fully_connect0',
                             residual = False,
                             padding = 'VALID',
                             act_fun = tf.nn.softplus,
                             bias = True)
        #hidden = self.conv2d(hidden, 1, 1, 1024, 
        #                     var_scope = 'fully_connect1',
        #                     residual = True,
        #                     act_fun = tf.nn.softplus,
        #                     bias = False)

        logits = self.conv2d(hidden, 1, 1, 10, 
                             var_scope = 'output',
                             residual = False,
                             act_fun = None,
                             dropout = False,
                             bias = True)
        
        self.logits = logits[:, 0, 0, :]
        self.target_one_hot = tf.one_hot(self.targets,
                                         depth = 10)
        self.loss_per_example = tf.nn.softmax_cross_entropy_with_logits(logits = self.logits,
                                                                        labels = self.target_one_hot)
        
        self.pred = tf.cast(tf.argmax(self.logits, 
                                      axis = 1),
                            self.targets.dtype)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred,
                                                        self.targets),
                                               tf.float32))

        
        # make sure the following attributes is defined
        assert hasattr(self, "inputs"), \
            "self.inputs undefined"
        assert hasattr(self, "targets"), \
            "self.targets undefined"
        assert hasattr(self, "loss_per_example"), \
            "self.loss_per_example undefined"
        
        