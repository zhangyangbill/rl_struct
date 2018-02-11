import copy
import itertools
import numpy as np
import tensorflow as tf

def dRNN(cell, inputs, rate, scope='default'):
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
    
    # make inputs a large tensor
    inputs = tf.stack(inputs, axis=1)
    
    batch_size, n_steps, input_dims = inputs.get_shape().as_list()
       
    # make the length of inputs divide 'rate', by using zero-padding
    #EVEN = tf.equal(tf.mod(n_steps, rate), 0)
    
    #def isEVEN():
    #    dilated_n_steps = tf.floordiv(n_steps, rate, name='isEVEN')
    #    inputs_ = tf.Print(inputs, [dilated_n_steps], "Input length for sub-RNN")
    #    return (dilated_n_steps, inputs_)
    #    
    #def notEVEN():
    #    dilated_n_steps = tf.floordiv(n_steps, rate, name='notEVEN') + 1
    #    #tf.Print(rate, [dilated_n_steps*rate-n_steps], "Time points need to be padded")
    #    #tf.Print(rate, [dilated_n_steps], "Input length for sub-RNN")
    #    paddings = [[0,0],[0, dilated_n_steps*rate-n_steps],[0,0]]
    #    inputs_ = tf.pad(inputs, paddings, constant_values=0.0)
    #    return (dilated_n_steps, inputs_) 
    
    dilated_n_steps = tf.cast(tf.ceil(tf.div(tf.cast(n_steps, tf.float32), 
                                     tf.cast(rate, tf.float32))), tf.int32)
    #paddings = tf.convert_to_tensor([[0,0],[0, dilated_n_steps*rate-n_steps],[0,0]])
    #inputs = tf.pad(inputs, paddings, constant_values=0.0)
    
    zero_tensor = tf.zeros([batch_size, dilated_n_steps*rate-n_steps, input_dims])
    
    inputs = tf.concat([inputs, zero_tensor], 1)
        
    #dilated_n_steps, inputs = tf.cond(EVEN, isEVEN, notEVEN)
    print inputs.get_shape()
            
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
    dilated_inputs = tf.transpose(tf.reshape(tf.transpose(inputs, [1,0,2]), [n_steps/rate, batch_size*rate, input_dims]), [1,0,2])
    print dilated_inputs.get_shape()
    splitting = tf.ones([1, n_steps/rate], tf.int32)
    dilated_inputs = list(tf.split(dilated_inputs, splitting, axis=1))
    print dilated_inputs[0].get_shape()
    
    dilated_inputs = [d[:, 0, :] for d in dilated_inputs]
    print dilated_inputs[0].get_shape()

    # building a dilated RNN with reformated (dilated) inputs
    dilated_outputs, _ = tf.contrib.rnn.static_rnn(
        cell, dilated_inputs,
        dtype=tf.float32, scope=scope)
    
    print dilated_outputs[0].get_shape()

    # reshape output back to the input format as a list of tensors with shape [batch_size, input_dims]
    # split each element of the outputs from size [batch_size*rate, input_dims] to 
    # [[batch_size, input_dims], [batch_size, input_dims], ...] with length = rate
    splitted_outputs = [tf.split(output, tf.fill([1, rate], 32), axis=0)
                        for output in dilated_outputs]
    unrolled_outputs = [output
                        for sublist in splitted_outputs for output in sublist]
    # remove padded zeros
    outputs = unrolled_outputs[:n_steps]

    return outputs



def multi_dRNN_with_dilations(cells, inputs, dilations):
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
    
    l = 0
    for cell, dilation in zip(cells, dilations):
        scope_name = "multi_dRNN_dilation_%d" % l
        x = dRNN(cell, x, dilation, scope=scope_name)
        l += 1
    return x
