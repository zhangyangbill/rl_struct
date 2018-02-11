import tensorflow as tf
import numpy as np
from pdrnn import triangular_pmf, build_pdrnn, compute_loss


class StochasticDilateNet:
    def __init__(self,
                 hidden_structs,
                 init_params,
                 n_layers,
                 n_steps,
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
        self.n_steps = n_steps
        self.n_classes = n_classes
        self.input_dims = input_dims
        self.cell_type = cell_type
        
        
        
        
        
        
        # build the network
        self.inputs = tf.placeholder(tf.float32, shape = [None, 20, 1], name = 'inputs')
        self.logits = drnn_classification(self.inputs,
                                          hidden_structs,
                                          dilations,
                                          n_steps,
                                          n_classes,
                                          input_dims,
                                          cell_type)
        
        # model weights
        self.weights = [v for v in tf.trainable_variables() if 'multi_dRNN_dilation' in v.name]
        self.struct_param = [v for v in tf.trainable_variables() if 'struct_layer' in v.name]
        
               
        # define modified loss for REINFORCE
        self.log_p_per_example \
        = [tf.log(triangular_pmf(dd, pp[0], pp[1])) for dd, pp in zip(self.rates, self.structs)]
        
        # multiply a phantom term log_p to compute gradient over struct_param
        self.loss_per_example = compute_loss(self.logits, self.labels)
        self.loss_for_w = tf.reduce_mean(self.loss_per_example)
        self.loss_for_pi = tf.reduce_mean(self.log_p_per_example * self.loss_per_example)

        
        # define optimizers
        self.weights_train_op = tf.train.AdamOptimizer()\
                                .minimize(self.loss_for_w,
                                          var_list = self.weights)
        
        self.struct_train_op = tf.train.AdamOptimizer()\
                                .minimize(self.loss_for_pi,
                                          var_list = self.struct_param)
        
        
        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver(max_to_keep=20)
        
        # define session
        self.sess = tf.Session()
        
        # initialize all the parameters
        self.sess.run(tf.global_variables_initializer())
        
        # optimizer counters
        self.weight_counter = 0
        self.train_step_counter = 0
        self.struct_counter = 0
        self.epoch_counter = 0
        
        
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
               bias = False,
               block_shape = [1, 1, 1]):
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
        bias - True if bias is applied
        block_shape - the shape of the output sub-tensor that shares the same pi
        
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
                pi_shape = [np.ceil(float(o) / float(b))
                           for o, b in zip(out_shape_list[1:], block_shape)]
                pi = tf.get_variable('drop_prob', 
                                     shape = pi_shape,
                                     initializer = tf.random_uniform_initializer(minval = self.min_init_dropout_rate,
                                                                                 maxval = self.max_init_dropout_rate))
                self.struct_param += [pi]
                #pi = tf.slice(tf_repeat(pi, block_shape), 
                #              [0,0,0], 
                #              out_shape_list[1:])
                #self.struct_param_tiled += [pi]

            # sample from Bernoulli distribution or from default
            keep_shape = tf.concat([out_shape[0:1], tf.shape(pi)], axis = 0)
            keep_shape_list = out_shape_list[0:1] + pi.get_shape().as_list()
            keep = tf.placeholder_with_default(
                  tf.where(tf.random_uniform(keep_shape) > pi,
                           tf.ones(keep_shape),
                           tf.zeros(keep_shape)),
                  shape = keep_shape_list)
            
            # stop the gradient to prevent NaN over pi
            keep = tf.stop_gradient(keep)
            self.keeps += [keep]
            keep = tf.slice(tf_repeat(keep, [1] + block_shape), 
                            [0,0,0,0], 
                            [-1] + out_shape_list[1:])
            print(keep)
            self.keeps_tiled += [keep]
            
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
        epoch_id = self.head
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
                loss, accuracy = self.sess.run([self.loss_true,
                                                self.accuracy],
                                               feed_dict = feed_dict)
                print('Epoch {}, weight train batch {}, step {}, loss = {}, accuracy = {}'.format(epoch_id,
                                                                                                  batch_id,
                                                                                                  w_id,
                                                                                                  loss,
                                                                                                  accuracy))
                # print struct param
                if batch_id % 100 == 0:
                    struct_param_value, valid_index = self.sess.run([self.struct_param, self.valid_index], feed_dict = feed_dict)
                    print('struct_param_value = {}'.format(struct_param_value))
                    print('valid_index = {}'.format(valid_index))
                    
                    accs = []
                    for i in range(10):
                        feed_dict = {self.inputs: inputs_test[i*1000:(i+1)*1000, ...], 
                                     self.targets: targets_test[i*1000:(i+1)*1000, ...]}
                        acc = self.sess.run(self.accuracy, feed_dict=feed_dict)
                        accs.append(acc)
                    print('Test accuracy ={}'.format(sum(accs)/10))
                    
                    
                    
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
                    save_path = self.saver.save(self.sess, "{}model_unroll".format(self.LOG_DIR), global_step=epoch_id)
                    print("Model saved in file: %s" % save_path)
                    
                    
                    
                if epoch_id >= num_epochs:
                    break
                    
            ###### perform struct training
            if epoch_id > 10:
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
                self.sess.run(self.struct_train_op,
                              feed_dict = feed_dict)
                
                #if struct_step_id % 100 == 0:
                #    struct_param_value = self.sess.run(self.struct_param, feed_dict = feed_dict)
                #    print('struct_param_value = {}'.format(struct_param_value))
                
                #bias_value = self.sess.run([self.weights[-1], self.weights[-3]], feed_dict = feed_dict)
                #print('bias_value = {}'.format(bias_value))
                
                self.sess.run(self.struct_clip_op,
                              feed_dict = feed_dict)
                
                
                
                # increment the counters
                struct_step_id += 1
                
                # ouput training information
                loss, accuracy, struct_param = self.sess.run([self.loss_true, 
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
        
        # define dataset
        mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
        inputs_test, targets_test = mnist.test.next_batch(10000)
        inputs_test = inputs_test.reshape((-1, 28, 28, 1))
        
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
                loss, accuracy = self.sess.run([self.loss_true,
                                                self.accuracy],
                                               feed_dict = feed_dict)
                print('Epoch {}, weight train batch {}, step {}, loss = {}, accuracy = {}'.format(epoch_id,
                                                                                                  batch_id,
                                                                                                  w_id,
                                                                                                  loss,
                                                                                                  accuracy))
                # print struct param
                if batch_id % 100 == 0:                 
                    accs = []
                    for i in range(10):
                        feed_dict = {self.inputs: inputs_test[i*1000:(i+1)*1000, ...], 
                                     self.targets: targets_test[i*1000:(i+1)*1000, ...]}
                        acc = self.sess.run(self.accuracy, feed_dict=feed_dict)
                        accs.append(acc)
                    print('Test accuracy = {}'.format(sum(accs)/10))
                    
                
   
                # increment the counters
                batch_id = batch_id + 1
                if batch_id >= num_train_batches:
                    # reshuffle the data
                    shuffle = shuffle_next
                    shuffle_next = np.random.permutation(num_tokens_train)
                    batch_id = 0
                    struct_step_id = 0
                    epoch_id += 1
                    save_path = self.saver.save(self.sess, "{}model_fine".format(self.LOG_DIR), global_step=epoch_id)
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
            loss, accuracy = self.sess.run([self.loss_true, 
                                            self.accuracy],
                                           feed_dict = feed_dict)
            print('Validation loss = {}, accuracy = {}'.format(loss,
                                                               accuracy))
    
    def build_graph(self):
        ''' This function builds the dropout net architecture'''
    
        self.inputs = tf.placeholder(tf.float32, shape = [None, 28, 28, 1], name = 'inputs')
        self.targets = tf.placeholder(tf.int32, shape = [None,], name = 'targets')
        
        hidden = self.conv2d(self.inputs, 5, 5, 64, 
                             var_scope = 'conv0',
                             residual = False,
                             padding = 'SAME',
                             act_fun = tf.nn.softplus,
                             bias = True,
                             block_shape = [4, 4, 8])
        hidden = self.conv2d(hidden, 5, 5, 64, 
                             var_scope = 'conv1',
                             residual = True,
                             padding = 'SAME',
                             act_fun = tf.nn.softplus,
                             bias = True,
                             block_shape = [4, 4, 8])
        hidden = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')
        
        hidden = self.conv2d(hidden, 5, 5, 128, 
                             var_scope = 'conv2',
                             residual = False,
                             padding = 'SAME',
                             act_fun = tf.nn.softplus,
                             bias = True,
                             block_shape = [2, 2, 16])

        hidden = self.conv2d(hidden, 5, 5, 128, 
                             var_scope = 'conv3',
                             residual = True,
                             padding = 'SAME',
                             act_fun = tf.nn.softplus,
                             bias = True,
                             block_shape = [2, 2, 16])
        hidden = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')

        
        hidden = self.conv2d(hidden, 7, 7, 2048,
                             var_scope = 'fully_connect0',
                             residual = False,
                             padding = 'VALID',
                             act_fun = tf.nn.softplus,
                             bias = True,
                             block_shape = [1, 1, 256])
        hidden = self.conv2d(hidden, 1, 1, 2048, 
                             var_scope = 'fully_connect1',
                             residual = True,
                             act_fun = tf.nn.softplus,
                             bias = True,
                             block_shape = [1, 1, 256])

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
        
        