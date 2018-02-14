import copy
import itertools
import numpy as np
import tensorflow as tf

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
    dilated_outputs, _ = tf.nn.dynamic_rnn(
        cell, dilated_inputs, time_major=True,
        dtype=tf.float32, scope=scope)
    

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
