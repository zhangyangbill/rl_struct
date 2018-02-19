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



def multi_dRNN_with_dilations(cells, inputs, hidden_structs, dilations):
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
    x = copy.copy(inputs)
    _, _, init_dim = x.get_shape().as_list()
    dims = [init_dim] + hidden_structs[:-1]
    l = 0
    for cell, dilation, dim in zip(cells, dilations, dims):
        scope_name = "multi_dRNN_layer_%d" % l
        x = dRNN(cell, x, dilation, dim, scope=scope_name)
        l += 1
    return x



def _contruct_cells(hidden_structs, cell_type):
    """
    This function contructs a list of cells.
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
                        input_dims=1,
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
    cells = _contruct_cells(hidden_structs, cell_type)

    # define dRNN structures
    layer_outputs = multi_dRNN_with_dilations(cells, x, hidden_structs, dilations)
    
    with tf.variable_scope('multi_dRNN_layer_final'):  
        # define the output layer
        h = tf.transpose(layer_outputs[-10:,:,:], perm=[1,0,2])
        pred = _conv1d(h,
                       in_channels=hidden_structs[-1],
                       out_channels=n_classes, 
                       filter_width=1, 
                       padding='valid', 
                       gain=1, 
                       activation=None,
                       bias=True,
                       trainable=True)
        # define the output layer
        #weights = tf.get_variable("w_conv", initializer=tf.random_normal(shape=[10, hidden_structs[-1], n_classes]))
        #bias = tf.get_variable("b_conv", initializer=tf.random_normal(shape=[n_classes]))
        # define prediction
    #pred = tf.add(tf.matmul(layer_outputs[-10:,:,:], weights), bias)

    return pred



def build_pdrnn(x,
                hidden_structs,
                init_params,
                n_layers,
                n_steps,
                n_classes,
                input_dims=1,
                cell_type="RNN"):
    
    # initialize struct params 
    params = []
    dilations = []
    for l in xrange(n_layers):
        with tf.variable_scope('struct_layer_{}'.format(l)):
            mu = tf.get_variable('mu', initializer=tf.constant(init_params[l][0]))
            sigma = tf.get_variable('sigma', initializer=tf.constant(init_params[l][1]))
            params.append((mu,sigma))
            
        rates = tf.range(tf.ceil(tf.maximum(mu-sigma,0.5)), 
                         tf.ceil(tf.minimum(mu+sigma,n_steps)))
        probs = triangular_pmf(rates, mu, sigma, n_steps)
        probs = tf.expand_dims(probs, 0)
        idx = tf.multinomial(tf.log(probs), 1)
        rates = tf.cast(rates, tf.int32)
        dilations.append(rates[idx[0][0]])  
        
    dilations[0] = tf.constant(1)
    
    logits = drnn_classification(x,
                                 hidden_structs,
                                 dilations,
                                 n_classes,
                                 input_dims,
                                 cell_type)
    
    return (logits, params, dilations)





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



    
    
    
    