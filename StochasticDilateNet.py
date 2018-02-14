import tensorflow as tf
import numpy as np
from pdrnn import triangular_pmf, build_pdrnn, compute_loss


class StochasticDilateNet:
    def __init__(self,
                 hidden_structs,
                 init_params,
                 n_layers,
                 n_classes,
                 input_dims=1,
                 lambda_b=0.9,
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
        self.labels = tf.placeholder(tf.int64, [None])
        
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
        self.struct_param = struct_param[2:] # keep bottom layer param constant
            
            
        # define modified loss for REINFORCE
        self.log_p_per_example \
        = sum([tf.log(triangular_pmf(dd, pp[0], pp[1], self.n_steps)) 
               for dd, pp in zip(self.rates, self.struct_vars)])
        
        
        # multiply a phantom term log_p to compute gradient over struct_param
        self.loss_per_example = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                labels=self.labels, logits=self.logits, name='cross_entropy_per_example')
        self.loss_for_w = tf.reduce_mean(self.loss_per_example)
        
        b = tf.get_variable('b', initializer=tf.constant(0.0))
        b = tf.assign(b, (lambda_b*b+(1-lambda_b)*self.loss_for_w))
        self.b = b
        self.loss_for_pi = tf.reduce_mean(self.log_p_per_example * (self.loss_per_example - b))
        
        # model evaluation
        correct_pred = tf.equal(tf.argmax(self.logits,1), self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        
        # define optimizers
        self.weights_train_op = tf.train.AdamOptimizer()\
                                .minimize(self.loss_for_w,
                                          var_list = self.weights)
        
        self.struct_train_op = tf.train.AdamOptimizer(learning_rate=0.1)\
                                .minimize(self.loss_for_pi,
                                          var_list = self.struct_param)
            
        # clip mu and sigma
        self.struct_clip_op = []
        for mu, sigma in self.struct_vars:
            mu_clipped = tf.clip_by_value(mu, 2, self.n_steps-1.5)
            sigma_clipped = tf.clip_by_value(sigma, 1.5, 
                                             tf.minimum(mu_clipped-0.5, self.n_steps-mu_clipped))
            structs_clipped = tf.group(tf.assign(mu,mu_clipped), 
                                       tf.assign(sigma,sigma_clipped))
            self.struct_clip_op.append(structs_clipped)
                    
        
        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver(max_to_keep=20)
            
                
   
                    
    def train(self,
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
    