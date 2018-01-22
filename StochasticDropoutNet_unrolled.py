import tensorflow as tf
import numpy as np
import random
from util import LBFGS, graph_replace, extract_update_dict

class StochasticDropoutNet:
    def __init__(self,
                 train_batch_size = 32,
                 valid_batch_size = 32,
                 num_weight_train_steps = 4,
                 weight_delta_max_count = 128,
                 min_dropout_rate = 0.01,
                 max_dropout_rate = 0.99,
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
        
        # define lists for weight delta and weight grad delta
        self.weight_delta = []
        self.weight_grad_delta = []
        self.weight_delta_max_count = weight_delta_max_count
        
        # build the network
        self.build_graph_test()
        
        # define modified loss for REINFORCE
        self.log_p_per_example \
        = sum([tf.reduce_sum(tf.log(pp) * (1 - kk) + tf.log(1 - pp) * kk,
                             axis = [1,2,3]) 
               for pp, kk in zip(self.struct_param, self.keeps)])
        self.log_p = tf.reduce_mean(self.log_p_per_example)
        # multiply a phantom term log_p to compute gradient over struct_param
        self.loss = tf.reduce_mean(self.loss_per_example * 
                                   (tf.stop_gradient(1 - self.log_p_per_example)
                                    + self.log_p_per_example))
        self.loss_modified = tf.reduce_mean(self.log_p_per_example * self.loss_per_example)
        
        # define optimizer & gradients
        self.optimizer_weight = tf.train.AdamOptimizer()
        self.optimizer_struct = tf.train.AdamOptimizer(learning_rate = 0.05)
        self.weight_gradients = tf.gradients(self.loss, self.weights)
        self.struct_gradients = tf.gradients(self.loss_modified, self.struct_param)
        self.log_p_grad = tf.gradients(self.log_p, self.struct_param)

        # define a placeholder for processed gradients
        self.weight_gradients_p = [tf.placeholder_with_default(g, shape = g.get_shape())
                                   for g in self.weight_gradients]
        self.compute_weight_gradients_p = list(zip(self.weight_gradients_p, self.weights))
        self.apply_weight_gradients = self.optimizer_weight.apply_gradients(self.compute_weight_gradients_p)
        
        self.struct_gradients_p = [tf.placeholder_with_default(g, shape = g.get_shape())
                                   for g in self.struct_gradients]
        self.compute_struct_gradients_p = list(zip(self.struct_gradients_p, self.struct_param))
        self.apply_struct_gradients = self.optimizer_struct.apply_gradients(self.compute_struct_gradients_p)
        
        # define unrolling parameters
        self.unroll_steps = unroll_steps
        self.inputs_adapt = [tf.placeholder_with_default(shape = self.inputs.shape,
                                                         input = self.inputs)
                             for i in xrange(self.unroll_steps)]
        self.targets_adapt = [tf.placeholder_with_default(shape = self.targets.shape,
                                                          input = self.targets)
                             for i in xrange(self.unroll_steps)]
        
        # define unrolling optimizers
        #### test!!! change back to Adam
        w_opt = tf.keras.optimizers.Adam(lr = 0.01)
        updates = w_opt.get_updates(self.loss, self.weights)
        self.weights_train_op = tf.group(*updates, name="weights_train_op")
        
        update_dict = extract_update_dict(updates)
        cur_update_dict = graph_replace(update_dict, 
                                        {self.inputs: self.inputs_adapt[0],
                                         self.targets: self.targets_adapt[0]})
        ##### test!!
        self.weights_adapt = [[cur_update_dict[w.value()] for w in self.weights]]
        for i in xrange(self.unroll_steps-1):
            # Change the inputs
            update_dict_adapt = graph_replace(update_dict, 
                                              {self.inputs: self.inputs_adapt[i+1],
                                               self.targets: self.targets_adapt[i+1]})
            # Compute variable updates given the previous iteration's updated variable
            cur_update_dict = graph_replace(update_dict_adapt, cur_update_dict)
            
            ###### test!!!
            self.weights_adapt.append([cur_update_dict[w.value()] for w in self.weights])
            
        # Final unrolled loss uses the parameters at the last time step
        self.unrolled_loss = graph_replace(self.loss, cur_update_dict)
        self.struct_train_op = tf.train.AdamOptimizer(learning_rate = 0.05)\
                                .minimize(self.unrolled_loss,
                                          var_list = self.struct_param)
        # clip the value
        self.struct_clip_op = []
        for var in self.struct_param:
            var_clipped = tf.clip_by_value(var, self.min_dropout_rate, self.max_dropout_rate)
            self.struct_clip_op.append(tf.assign(var, var_clipped))
        
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
            w = tf.get_variable('weights', 
                                [filter_height, filter_width, num_in_filters, num_filters])
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
                b = tf.get_variable('biases',
                                    shape = out_shape_list[1:])
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
    
    def _weights_train_step(self,
                           train_inputs,
                           train_targets,
                           previous_gradient = None):
        '''
        This function performs one update step update of the weight training,
        and returns the weight changes, as well as gradient value after the step.
        If no previous gradient is provided, the update will not be preformed.
        
        
        Args:
        train_data - a numpy array of training input data
        train_targets - a numpy array of training targets
        previous_gradient - a list of tuples of gradients computed on last minibatch of data
        
        Returns:
        weight_delta - the change in weights after the train step.
                       If previous_gradient = None, weight_delta = None.
        grad_before - the gradient before the gradient update on the new data
        grad_after - the gradient before the gradient update on the new data
        loss_before - the loss before the gradient update on the new data
        loss_after - the loss after the gradient update on the new data
        '''
        
        
        # compute the gradients, losses and keep notes
        grad_before, loss_before, keeps, weights_before \
        = self.sess.run([self.weight_gradients, 
                         self.loss, 
                         self.keeps, 
                         self.weights],
                        feed_dict = {self.inputs: train_inputs,
                                     self.targets: train_targets})
        
            
        if previous_gradient is not None:
            # modify gradients by averaging the previous gradient
            grad_p = [(g1 + g2) / 2 for g1, g2 in zip(grad_before, previous_gradient)]

            # apply gradient
            feed_dict = dict(zip([self.inputs, self.targets]
                                 + self.weight_gradients_p,
                                 [train_inputs, train_targets] 
                                 + grad_p))
            
            self.sess.run(self.apply_weight_gradients,
                          feed_dict = feed_dict)
            
            # evaluate loss and gradient after the update
            grad_after, loss_after, weights_after, summary\
            = self.sess.run([self.weight_gradients, 
                             self.loss, 
                             self.weights, 
                             self.merged_summary],
                            feed_dict = dict(zip([self.inputs, self.targets]
                                                 + self.weight_gradients_p
                                                 + self.keeps,
                                                 [train_inputs, train_targets] 
                                                 + grad_p
                                                 + keeps)))
            
            # write tensorboard summary and losses
            self.train_writer.add_summary(summary, self.train_step_counter)
            self.train_step_counter += 1
            self.train_losses.append(loss_after)
            
            # find out the weight difference
            weight_delta = [w1 - w2 for w1, w2 in zip(weights_after, weights_before)]
        
        else:
            grad_after = None
            loss_after = None
            weight_delta = None
        
        return weight_delta, grad_before, grad_after, loss_before, loss_after
    
    def weights_train_batch(self,
                            train_inputs,
                            train_targets):
        '''
        This function performs a batch of weight train step, while recording information for L-BFGS
        
        Args:
        train_input - a numpy array of training input data, 1st dim is token
        train_targets - a numpy array of training targets, 1st dim is token
        
        Writes:
        self.weight_delta - a list of weight changes in the previous weight updates
        self.weight_grad_delta - a list of weight gradient changes in the previous weight updates
        
        Returns:
        weight_grad_per_example - the per example weight gradient of the final model
        log_p_grad_final - the per sample gradient of log p over structual parameters
        '''
        
        # number of tokens
        num_tokens = train_inputs.shape[0]
        
        # number of batches
        num_batches = np.floor(num_tokens / self.train_batch_size)
        
        # initialize
        # shuffle data
        shuffle = np.random.permutation(num_tokens)
        batch_id = 0
        # initially evaluate gradient
        _, gb, _, _, _ = self._weights_train_step(train_inputs[shuffle[batch_id * self.train_batch_size :
                                                                       (batch_id+1) * self.train_batch_size],
                                                               ...],
                                                  train_targets[shuffle[batch_id * self.train_batch_size :
                                                                        (batch_id+1) * self.train_batch_size],
                                                                ...])
        batch_id += 1
        
        
        # iteratively call train step
        for it in xrange(self.num_weight_train_steps):
            if batch_id >= num_batches:
                # reshuffle data
                shuffle = np.random.permutation(num_tokens)
                batch_id = 0
                
            # call train step
            wd, gb, ga, lb, la = self._weights_train_step(train_inputs[shuffle[batch_id * self.train_batch_size :
                                                                             (batch_id+1) * self.train_batch_size],
                                                                     ...],
                                                        train_targets[shuffle[batch_id * self.train_batch_size :
                                                                              (batch_id+1) * self.train_batch_size],
                                                                      ...],
                                                        previous_gradient = gb)
            # output information
            print('Epoch:{}, weight train batch: {}, step:{}, loss before: {}, loss after: {}.'
                  .format(self.epoch_counter, self.weight_counter, it, lb, la))
            
            # record information for L-BFGS
            if len(self.weight_delta) >= self.weight_delta_max_count:
                self.weight_delta.pop(0) # maintain the queue length
                self.weight_grad_delta.pop(0)
            self.weight_delta += [wd]
            self.weight_grad_delta += [[_ga - _gb for _gb, _ga in zip(gb, ga)]]
            
            # increment counter
            batch_id += 1
        
        # evaluate the per example gradient on the final model using the final batch data
        final_batch_idx = (batch_id-1) * self.train_batch_size
        weight_grad_per_example, log_p_grad_final \
        = list(zip(*[self.sess.run([self.weight_gradients, 
                                    self.log_p_grad],
                                   feed_dict = {self.inputs: train_inputs[shuffle[final_batch_idx + i :
                                                                                  final_batch_idx + i + 1]],
                                                self.targets: train_targets[shuffle[final_batch_idx + i :
                                                                                    final_batch_idx + i + 1]]})
                     for i in xrange(self.train_batch_size)]))
        
        # increment global counter
        self.weight_counter += 1
        
        return weight_grad_per_example, log_p_grad_final
    
    def struct_train_step(self, 
                          weight_grad_per_example, 
                          log_p_grad, 
                          valid_inputs, 
                          valid_targets,
                          clip_min = 0.001,
                          clip_max = 0.999):
        ''' This function preforms one gradient update step for structual parameters.
        
        Args:
        weight_delta - a list of weight changes in the previous weight updates
        weight_grad_delta - a list of weight gradient changes in the previous weight updates
        weight_grad_per_example - the per example weight gradient of the final model
        log_p_grad - the per sample gradient of log p over structual parameters
        valid_inputs - a numpy array of validation input data, 1st dim is token
        valid_targets - a numpy array of training targets, 1st dim is token
        [clip_min, clip_max] - defines the interval to clip the structural parameters
        '''
                
        ### compute the second term of the gradient
        
        # compute d/d\theta g and the second term of the gradient
        weight_grad_valid, _struct_grad2, loss_before, keeps \
        = self.sess.run([self.weight_gradients,
                         self.struct_gradients,
                         self.loss, 
                         self.keeps],
                        feed_dict = {self.inputs: valid_inputs,
                                     self.targets: valid_targets})
        
        
        ### compute the first term of the gradient
        
        # compute H_theta f \ d/d\theta g using LBFGS
        _H_struct_grad = LBFGS(self.weight_delta,
                               self.weight_grad_delta,
                               weight_grad_valid)
        
        # multiplied with H_{theta, pi} f to get the first term in the struct gradient
        _d_theta_f_H_struct_grad = [sum([np.sum(ww * hh) 
                                         for ww, hh in zip(weight_grad_per_example[i], _H_struct_grad)]) 
                                    for i in xrange(self.train_batch_size)]
        _struct_grad1 = [sum([log_p_grad[i][j] * _d_theta_f_H_struct_grad[i]
                             for i in xrange(self.train_batch_size)]) / self.train_batch_size
                         for j in xrange(len(log_p_grad[0]))]
        
        # compute gradient mask, if a node has too unbalanced samples, set the gradient to 0
        keep_ratio = [np.sum(k, axis = 0).astype(float) / k.shape[0] 
                      for k in keeps]
        grad_mask = [np.where(np.logical_and(kr > 0.01, kr < 0.99), 
                              np.ones(kr.shape), 
                              np.zeros(kr.shape))
                     for kr in keep_ratio]
                
        ### add the two gradient terms
        struct_grad = [g2 - g1 
                       for g1, g2 in zip(_struct_grad1,
                                         _struct_grad2)]
        
        # apply the gradient
        feed_dict = dict(zip([self.inputs, self.targets]
                             + self.struct_gradients_p,
                             [valid_inputs, valid_targets] 
                             + struct_grad))
        
        #self.sess.run(self.apply_struct_gradients,
        #              feed_dict = feed_dict) ### recover the comments
        
        # clip the variable to [clip_min, clip_max]
        for i in xrange(len(self.struct_param)):
            var_clipped = tf.clip_by_value(self.struct_param[i], clip_min, clip_max)
            print(self.sess.run(var_clipped))
            self.sess.run(tf.assign(self.struct_param[i], var_clipped))
        
        # evaluate loss and gradient after the update
        loss_after = self.sess.run(self.loss,
                                   feed_dict = feed_dict)
        
        # write losses
        self.valid_losses.append(loss_after)
        
        # output information
        print('Epoch:{}, struct parameters train batch: {}, loss before: {}, loss after: {}.'
              .format(self.epoch_counter, self.struct_counter, loss_before, loss_after))
        
        # increment global counter
        self.struct_counter += 1
        
    def train(self,
              train_inputs,
              train_targets,
              valid_inputs,
              valid_targets, 
              num_epochs = 10):
        '''
        This function performs the training process by iteratively calling weight train and struct train functions.
        
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
        array_num_tokens_valid = range(num_tokens_valid)
        
        # determine batch size
        train_batch_size = self.train_batch_size * self.num_weight_train_steps
        
        # determine number of batches
        num_batches_train = num_tokens_train / train_batch_size
        # num_batches_valid = num_tokens_valid / self.valid_batch_size
        
        # iterate epochs
        for epoch_id in xrange(num_epochs):
            # shuffle the data
            shuffle = np.random.permutation(num_tokens_train)
            self.epoch_counter = epoch_id
            for batch_id in xrange(num_batches_train):
                # weights updates
                
                selected_tokens_train = shuffle[batch_id * train_batch_size :
                                                (batch_id+1) * train_batch_size]
                wg_pe, lpg_f \
                = self.weights_train_batch(train_inputs[selected_tokens_train,...], 
                                           train_targets[selected_tokens_train,...])
                
                # structural parameters updates
                if self.weight_counter > 4: # allow some burn-in
                    selected_tokens_valid = random.sample(array_num_tokens_valid, 
                                                          self.valid_batch_size)
                    self.struct_train_step(wg_pe, lpg_f,
                                           valid_inputs[selected_tokens_valid,...],
                                           valid_targets[selected_tokens_valid,...])
                    
                    
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
                
                # increment the counters
                batch_id = batch_id + 1
                if batch_id >= num_train_batches:
                    # reshuffle the data
                    shuffle = shuffle_next
                    shuffle_next = np.random.permutation(num_tokens_train)
                    batch_id = 0
                    struct_step_id = 0
                    epoch_id += 1
                    
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
            self.sess.run(self.struct_train_op,
                          feed_dict = feed_dict)
            self.sess.run(self.struct_clip_op,
                          feed_dict = feed_dict)
            
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
        
        hidden = self.conv2d(self.inputs, 5, 5, 128, 
                             var_scope = 'conv0',
                             residual = False,
                             padding = 'SAME',
                             act_fun = tf.nn.softplus)
        hidden = self.conv2d(hidden, 5, 5, 128, 
                             var_scope = 'conv1',
                             residual = True,
                             padding = 'SAME',
                             act_fun = tf.nn.softplus)
        hidden = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')
        
        hidden = self.conv2d(hidden, 5, 5, 128, 
                             var_scope = 'conv2',
                             residual = True,
                             padding = 'SAME',
                             act_fun = tf.nn.softplus)
        hidden = self.conv2d(hidden, 5, 5, 128, 
                             var_scope = 'conv3',
                             residual = True,
                             padding = 'SAME',
                             act_fun = tf.nn.softplus)
        hidden = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')
        
        hidden = self.conv2d(hidden, 7, 7, 2048, 
                             var_scope = 'fully_connect0',
                             residual = False,
                             padding = 'VALID',
                             act_fun = tf.nn.softplus,
                             bias = True)
        hidden = self.conv2d(hidden, 1, 1, 2048, 
                             var_scope = 'fully_connect1',
                             residual = True,
                             act_fun = tf.nn.softplus,
                             bias = True)
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
        
    def build_graph_test(self):
        self.inputs = tf.placeholder(tf.float32, shape = [None, 1, 1, 1], name = 'inputs')
        self.targets = tf.placeholder(tf.float32, shape = [None, 1, 1, 1], name = 'targets')
        
        hidden = self.conv2d(self.inputs, 1, 1, 1,
                             var_scope = 'conv0',
                             residual = True,
                             padding = 'VALID',
                             act_fun = None)
        outputs = self.conv2d(hidden, 1, 1, 1,
                              var_scope = 'output',
                              residual = False,
                              padding = 'VALID',
                              dropout = False,
                              act_fun = None)
        
        self.loss_per_example = tf.square(outputs[:, 0, 0, 0])