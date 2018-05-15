import copy
import itertools
import tensorflow as tf
import numpy as np


def ng_ceil(x):
    return tf.stop_gradient(tf.ceil(x))
def ng_floor(x):
    return tf.stop_gradient(tf.floor(x))


def triangular_pmf(x,
                   mu=4.0, 
                   sigma=3.0,
                   N=10**9):
    
    x = tf.cast(x, tf.float32)
    # compute normalizing constant
    L = (ng_ceil(tf.maximum(mu-sigma,0.5))+ng_floor(mu)-2*mu) / (2*sigma)
    R = (2*mu-ng_floor(mu+1)-ng_floor(tf.minimum(mu+sigma,N))) / (2*sigma)
    
    h_inv = (ng_floor(mu)-ng_ceil(tf.maximum(mu-sigma,0.5))+1)*(L+1)\
          + (ng_floor(tf.minimum(mu+sigma,N))-ng_floor(mu))*(R+1)
        
    h = 1.0 / h_inv    
           
    # define pmf
    y = -h/sigma * tf.abs(x-mu) + h
    
    return y



def general_logpmf(x, logits):
    log_probs = tf.nn.log_softmax(logits)
    
    return log_probs[0][x]
    
    
def pmf_entropy(logits):
    probs = tf.nn.softmax(logits)
    log_probs = tf.nn.log_softmax(logits)
    
    return -tf.reduce_sum(probs*log_probs)



def dRNN(cell, inputs, rate, dim, scope='default'):
    """
    This function constructs a layer of dilated RNN.
    Inputs:
        cell -- the dilation operations is implemented independent of the RNN cell.
            In theory, any valid tensorflow rnn cell should work.
        inputs -- the input for the RNN. inputs should be in the form of
            a list of 'n_steps' tenosrs. Each has shape (batch_size, input_dims)
        rate -- the rate here refers to the 'dilations' in the orginal WaveNet paper. 
        scope -- variable scope.
    Outputs:
        outputs -- the outputs from the RNN.
    """
        
    inputs_shape = tf.shape(inputs)
    n_steps = inputs_shape[0]
    batch_size = inputs_shape[1]
    input_dims = inputs_shape[2]
       
        
    dilated_n_steps = tf.cast(tf.ceil(tf.div(tf.cast(n_steps, tf.float32), 
                                     tf.cast(rate, tf.float32))), tf.int32)
    
    zero_tensor = tf.zeros([dilated_n_steps*rate-n_steps, batch_size, input_dims])
    
    inputs = tf.concat([inputs, zero_tensor], 0)
                    
    # now the length of 'inputs' divide rate
    # reshape it in the format of a list of tensors
    # the length of the list is 'dilated_n_steps' 
    # the shape of each tensor is [batch_size * rate, input_dims] 
    # by stacking tensors that "colored" the same

    # Example: 
    # n_steps is 5, rate is 2, inputs = [x1, x2, x3, x4, x5]
    # zero-padding --> [x1, x2, x3, x4, x5, 0]
    # we want to have --> [[x1; x2], [x3; x4], [x_5; 0]]
    # which the length is the ceiling of n_steps/rate
    
    dilated_inputs = tf.reshape(inputs, [dilated_n_steps, batch_size*rate, input_dims])
    dilated_inputs.set_shape([None, None, dim])
    
    # building a dilated RNN with reformated (dilated) inputs
    dilated_outputs, _ = tf.nn.dynamic_rnn(cell, 
                                           dilated_inputs, 
                                           time_major=True,
                                           dtype=tf.float32,
                                           parallel_iterations=64,
                                           scope=scope)
    

    # reshape output back to the input format as a list of tensors with shape [batch_size, input_dims]
    # split each element of the outputs from size [batch_size*rate, input_dims] to 
    # [[batch_size, input_dims], [batch_size, input_dims], ...] with length = rate
    unrolled_outputs = tf.reshape(dilated_outputs, [dilated_n_steps*rate, batch_size, cell.output_size])
    
    # remove padded zeros
    outputs = unrolled_outputs[:n_steps,:,:]
    
    return outputs



def multi_dRNN_with_dilations(cells, inputs, hidden_structs, dilations, dropout=False):
    """
    This function constucts a multi-layer dilated RNN. 
    Inputs:
        cells -- A list of RNN cells.
        inputs -- A list of 'n_steps' tensors, each has shape (batch_size, input_dims).
        dilations -- A list of integers with the same length of 'cells' indicates the dilations for each layer.
    Outputs:
        x -- A list of 'n_steps' tensors, as the outputs for the top layer of the multi-dRNN.
    """
    assert (len(cells) == len(dilations))
    hs = []
    x = copy.copy(inputs)
    hs.append(x)
    _, _, init_dim = x.get_shape().as_list()
    dims = [init_dim] + hidden_structs[:-1]
    for l, (cell, dilation, dim) in enumerate(zip(cells, dilations, dims)):
        scope_name = "multi_dRNN_layer_%d" % l
        x = dRNN(cell, x, dilation, dim, scope=scope_name)
        hs.append(x)
    return x, hs



def _construct_cells(hidden_structs, cell_type):
    """
    This function constructs a list of cells.
    """
    # error checking
    if cell_type not in ["RNN", "LSTM", "GRU"]:
        raise ValueError("The cell type is not currently supported.")

    # define cells
    cells = []
    for hidden_dims in hidden_structs:
        if cell_type == "RNN":
            cell = tf.contrib.rnn.BasicRNNCell(hidden_dims)
        elif cell_type == "LSTM":
            cell = tf.contrib.rnn.BasicLSTMCell(hidden_dims)
        elif cell_type == "GRU":
            cell = tf.contrib.rnn.GRUCell(hidden_dims)
        cells.append(cell)

    return cells



def _conv1d(inputs, 
            in_channels,
            out_channels, 
            filter_width=2, 
            stride=1, 
            padding='valid',
            data_format='channels_last',
            dilation_rate=1,
            gain=np.sqrt(2), 
            activation=None, 
            bias=True,
            name='',
            trainable=True):
     
    stddev = gain / np.sqrt(filter_width ** 2 * in_channels)
    w_init = tf.random_normal_initializer(stddev=stddev)      #weight matrix init
        
    outputs = tf.layers.conv1d(inputs=inputs, 
                               filters=out_channels, 
                               kernel_size=filter_width, 
                               strides=1, 
                               padding=padding, 
                               data_format=data_format, 
                               dilation_rate=dilation_rate, 
                               activation=activation, 
                               use_bias=bias, 
                               kernel_initializer=w_init, 
                               bias_initializer=w_init, 
                               kernel_regularizer=None, 
                               bias_regularizer=None, 
                               activity_regularizer=None, 
                               trainable=trainable, 
                               name=name, 
                               reuse=None)
       
    return outputs


def drnn_classification(x,
                        hidden_structs,
                        dilations,
                        n_classes,
                        n_evaluate,
                        dropout=False,
                        cell_type="RNN"):
    """
    This function construct a multilayer dilated RNN for classifiction.  
    Inputs:
        x -- a tensor of shape (batch_size, n_steps, input_dims).
        hidden_structs -- a list, each element indicates the hidden node dimension of each layer.
        dilations -- a list, each element indicates the dilation of each layer.
        n_steps -- the length of the sequence.
        n_classes -- the number of classes for the classification.
        input_dims -- the input dimension.
        cell_type -- the type of the RNN cell, should be in ["RNN", "LSTM", "GRU"].
    
    Outputs:
        pred -- the prediction logits at the last timestamp and the last layer of the RNN.
                'pred' does not pass any output activation functions.
    """
    # error checking
    assert (len(hidden_structs) == len(dilations))

    # construct a list of cells
    cells = _construct_cells(hidden_structs, cell_type)

    # define dRNN structures
    layer_outputs, test_out = multi_dRNN_with_dilations(cells, x, hidden_structs, dilations, dropout)
    
    with tf.variable_scope('multi_dRNN_layer_final'): 
        #h = tf.transpose(layer_outputs[-4:,:,:], perm=[1,0,2])
        #h = _conv1d(h,
        #            in_channels=hidden_structs[-1],
        #            out_channels=hidden_structs[-1], 
        #            filter_width=4, 
        #            padding='valid', 
        #            gain=np.sqrt(2), 
        #            activation=tf.tanh,
        #            bias=True,
        #            trainable=True)
        
        # define the output layer
        #h = tf.transpose(layer_outputs[-n_evaluate:,:,:], perm=[1,0,2])
        #pred = _conv1d(layer_outputs[-n_evaluate:,:,:],
        #               in_channels=hidden_structs[-1],
        #               out_channels=n_classes, 
        #               filter_width=1, 
        #               padding='valid', 
        #               gain=1, 
        #               activation=None,
        #               bias=True,
        #               trainable=True)
        
        weights = tf.Variable(tf.random_normal(shape=[hidden_structs[-1], n_classes]))
        bias = tf.Variable(tf.random_normal(shape=[n_classes]))
        # define prediction
        pred = tf.add(tf.matmul(layer_outputs[-1,:,:], weights), bias)

    return pred, test_out



def set_dilations_1(n_actions, n_layers):
    
    # initialize struct params 
    params = []
    picks = []
    for l in xrange(n_layers):
        with tf.variable_scope('struct_layer_{}'.format(l)):
            logpmf = tf.get_variable('logpmf', shape=[1,n_actions[l]],
                                     initializer=tf.constant_initializer(-3.0))
            params.append(logpmf)
            
        idx = tf.cast(tf.multinomial(logpmf, 1), tf.int32)
        picks.append(idx[0][0])
                
    return (picks, params) 



def multilayer_perceptron(x):
    
    w_1 = tf.get_variable('mlp_w_1', shape=[100,10],
                          initializer=tf.initializers.random_normal(stddev=0.01))
    b_1 = tf.get_variable('mlp_b_1', shape=[10],
                          initializer=tf.initializers.random_normal(stddev=0.01))
    hidden_layer = tf.add(tf.matmul(x, w_1), b_1)
    
    w_2 = tf.get_variable('mlp_w_2', shape=[10,510],
                          initializer=tf.constant_initializer(0.0))
    b_2 = tf.get_variable('mlp_b_2', shape=[510],
                          initializer=tf.constant_initializer(1.0/510))
    output_layer = tf.add(tf.matmul(hidden_layer, w_2), b_2)
    
    return output_layer


def set_dilations_2(support, n_layers):
    
    # initialize struct params 
    params = []
    picks = []
    dilations = []
    for l in xrange(n_layers):
        with tf.variable_scope('struct_layer_{}'.format(l)):
            if l == 0:
                logpmf = tf.get_variable('logpmf', shape=[1,support[l][1]],
                                         initializer=tf.constant_initializer(-0.01))
            else:
                logpmf = multilayer_perceptron(logpmf)
            params.append(logpmf)
            
        rates = tf.range(support[l][0], support[l][1]+1)  #add 1 to achieve upper limit
        idx = tf.cast(tf.multinomial(logpmf, 1), tf.int32)
        picks.append(idx[0][0])
        dilations.append(rates[idx[0][0]])  
        
    return (picks, dilations, params) 




def set_dilations_3(n_actions, n_layers, hidden_dim=100):
    
    empty_inputs = [tf.zeros([1,1]) for _ in xrange(n_layers+1)]
    
    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
    outputs, _ = tf.nn.static_rnn(cell, 
                                  empty_inputs, 
                                  dtype=tf.float32,
                                  scope='struct_layer')
    # convert to tensor
    h = tf.stack(outputs, axis=1)
    
    with tf.variable_scope('struct_layer'):
        logpmfs = tf.layers.conv1d(inputs=h[:,1:,:],
                                   filters=max(n_actions),
                                   kernel_size=1,
                                   kernel_initializer=tf.zeros_initializer(),
                                   bias_initializer=tf.constant_initializer(0.001))
    
    # initialize struct params 
    params = []
    picks = []
    for l in xrange(n_layers):
        logpmf = logpmfs[:, l, :n_actions[l]]
        params.append(logpmf)
        idx = tf.cast(tf.multinomial(logpmf, 1), tf.int32)
        picks.append(idx[0][0])
        
    return (picks, params) 




def set_dilations_4(n_actions, n_layers):
    
    # initialize struct params 
    params = []
    picks = []
    for l in xrange(n_layers):
        with tf.variable_scope('struct_layer_{}'.format(l)):
            logpmf = tf.get_variable('logpmf', shape=[1,n_actions[l]+9,1],
                                     initializer=tf.constant_initializer(-0.5))
            kernel = tf.contrib.signal.hamming_window(10, False)
            kernel = tf.expand_dims(tf.expand_dims(kernel,1),1)
            logpmf = tf.nn.conv1d(logpmf, kernel, 1, 'VALID')
            params.append(logpmf[:,:,0])
            
        idx = tf.cast(tf.multinomial(logpmf[:,:,0], 1), tf.int32)
        picks.append(idx[0][0])
                
    return (picks, params) 



def make_kernel(supports, n_actions):
    
    connections = np.array([1.0/4, 1.0/3, 1.0/2, 1.0, 2.0, 3.0, 4.0])  
    # [n_actions, n_connections], each row is one dilation rate copied n_connections times
    out_nodes = np.tile(np.arange(supports[0],supports[1]+1), 
                        (connections.shape[0],1)).T
    # multiple and fractions of each possible dilation rate
    candidates = connections * out_nodes
    # only integer valued dilation rate, remove out of range rates
    connectivity = np.equal(np.mod(candidates, 1), 0)
    partners = (candidates * connectivity).astype(int)
    partners = partners * (partners<=supports[1])   #actual rates
    
    hamm = np.array([0.08, 0.31, 0.77, 1.0, 0.77, 0.31, 0.08])
    hamm_connect = np.tile(hamm, (n_actions,1)) * (partners>0)
    values = hamm_connect / np.sum(hamm_connect, 1, keepdims=True) * 3.0
    values = values.flatten()
    values = values[values!=0].astype(np.float32)
    
    # map dilations to pmf index using 1-d array
    mapping = -np.ones(supports[1]+1, dtype=int)
    mapping[supports[0]:supports[1]+1] = np.arange(n_actions)
    # append row indices to pmf indices in sparse kernel
    real_idx = np.vstack((np.tile(np.arange(n_actions), 
                                  (connections.shape[0],1)).flatten('F'), 
                          mapping[partners].flatten())).T
    real_idx = real_idx[real_idx[:,1]!=-1]       # remove unused indices

    kernel = tf.sparse_to_dense(real_idx, [n_actions, n_actions], values)
    
    return kernel


def set_dilations_5(supports, n_actions, n_layers):
    
    # initialize struct params 
    params = []
    picks = []
    for l in xrange(n_layers):
        with tf.variable_scope('struct_layer_{}'.format(l)):
            logpmf = tf.get_variable('logpmf', shape=[n_actions[l],1],
                                     initializer=tf.constant_initializer(-1.0))
            kernel = make_kernel(supports[l], n_actions[l])
            logpmf = tf.sparse_matmul(kernel, logpmf, a_is_sparse=True)
            logpmf = tf.transpose(logpmf)
            params.append(logpmf)
            
        idx = tf.cast(tf.multinomial(logpmf, 1), tf.int32)
        picks.append(idx[0][0])
                
    return (picks, params) 




def make_mixed_kernel(supports, n_actions):
    
    connections = np.array([1.0/3, 1.0/2, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0])
    out_nodes = np.tile(np.arange(supports[0],supports[1]+1), 
                        (connections.shape[0],1)).T
    candidates = connections * out_nodes
    connectivity = np.equal(np.mod(candidates, 1), 0)
    partners = (candidates * connectivity).astype(int)
    partners = partners + np.tile(np.array([[0,0,-2,-1,0,1,2,0,0]]),(n_actions,1))
    partners = partners * ((partners<=supports[1]) & (partners>=supports[0]))
    
    hamm = np.hamming(connections.shape[0])
    hamm_connect = np.tile(hamm, (n_actions,1)) * (partners>0)
    # 1
    hamm_connect[0,5] += hamm_connect[0,7]
    hamm_connect[0,7] = 0.0
    hamm_connect[0,6] += hamm_connect[0,8]
    hamm_connect[0,8] = 0.0
    partners[0,[7,8]] = 0
    # 2
    hamm_connect[1,3] += hamm_connect[1,1]
    hamm_connect[1,1] = 0.0
    hamm_connect[1,6] += hamm_connect[1,7]
    hamm_connect[1,7] = 0.0
    partners[1,[1,7]] = 0
    # 3
    hamm_connect[2,2] += hamm_connect[2,0]
    hamm_connect[2,0] = 0.0
    partners[2,0] = 0
    # 4
    hamm_connect[3,2] += hamm_connect[3,1]
    hamm_connect[3,1] = 0.0
    partners[3,1] = 0
    
    values = hamm_connect / np.sum(hamm_connect, 1, keepdims=True) * 3.0
    values = values.flatten()
    values = values[values!=0].astype(np.float32)
    
    # map dilations to pmf index using 1-d array
    mapping = -np.ones(supports[1]+1, dtype=int)
    mapping[supports[0]:supports[1]+1] = np.arange(n_actions)
    # append row indices to pmf indices in sparse kernel
    real_idx = np.vstack((np.tile(np.arange(n_actions), 
                                  (connections.shape[0],1)).flatten('F'), 
                          mapping[partners].flatten())).T
    real_idx = real_idx[real_idx[:,1]!=-1]       # remove unused indices

    kernel = tf.sparse_to_dense(real_idx, [n_actions, n_actions], values)
    
    return kernel




def set_dilations_6(supports, n_actions, n_layers):
    
    # initialize struct params 
    params = []
    picks = []
    for l in xrange(n_layers):
        with tf.variable_scope('struct_layer_{}'.format(l)):
            logpmf = tf.get_variable('logpmf', shape=[n_actions[l],1],
                                     initializer=tf.constant_initializer(-1.0))
            kernel = make_mixed_kernel(supports[l], n_actions[l])
            logpmf = tf.sparse_matmul(kernel, logpmf, a_is_sparse=True)
            logpmf = tf.transpose(logpmf)
            params.append(logpmf)
            
        idx = tf.cast(tf.multinomial(logpmf, 1), tf.int32)
        picks.append(idx[0][0])
                
    return (picks, params) 






def compute_loss(logits, labels):
    """Compute total loss
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]
    Returns:
      Loss tensor of type float.
    """
    #print(logits.shape)
    #print(labels.shape)
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int32)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
  
    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss) if applicable.
    return tf.add_n(tf.get_collection('losses'), name='total_loss') 



    
def list_enqueue(x, lst, L=10):
    if len(lst) == L:
        lst.pop(0)
        lst.append(x)
    else:
        lst.append(x)
        
def find_closest_element(myNumber, lst):
    a = min(lst[::-1], key=lambda x:abs(x-myNumber))
    k = len(lst)-1-lst[::-1].index(a)
    return k

def list_enqueue_batch(xs, lsts):
    for i, x in enumerate(xs):
        list_enqueue(x, lsts[i])
        
def find_closest_element_batch(xs, lsts):
    ks = []
    for i, x in enumerate(xs):
        k = find_closest_element(x, lsts[i])
        ks.append(k)
    return ks   
 
        
        
def make_noisy_mnist(X, Y, T):
    X = np.expand_dims(np.transpose(X), axis=2)
    noisy = np.random.uniform(size=(T-784,X.shape[1],1))
    return (np.concatenate((X, noisy), axis=0), Y[np.newaxis,:])        
    
    