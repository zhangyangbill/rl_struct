import copy
import itertools
import tensorflow as tf
import numpy as np
from classification_models import drnn_classification


def triangular_pmf(x,
                   mu=4.0, 
                   sigma=3.0,
                   N=10**9):
    
    x = tf.cast(x, tf.float32)
    # compute normalizing constant
    L = (tf.ceil(tf.maximum(mu-sigma,0.5))+tf.floor(mu)-2*mu) / (2*sigma)
    R = (2*mu-tf.floor(mu+1)-tf.floor(tf.minimum(mu+sigma,N))) / (2*sigma)
    
    h_inv = (tf.floor(mu)-tf.floor(tf.maximum(mu-sigma,0.5)))*(L+1)\
          + (tf.floor(tf.minimum(mu+sigma,N))-tf.floor(mu))*(R+1)
    h = tf.stop_gradient(1 / h_inv)
       
    # define pmf
    y = -h/sigma * tf.abs(x-mu) + h
    
    return y


def sample_rate(n_layers):
    
    # sample dilation rate from distribution
    dilations = []
    for l in xrange(n_layers):
        rates = tf.range(tf.ceil(mu-sigma), mu+sigma)
        probs = triangular_pmf(rates, mu, sigma)
        probs = tf.expand_dims(probs, 0)
        idx = tf.multinomial(tf.log(probs), 1)
        rates = tf.cast(rates, tf.int32)
        dilations.append(rates[idx[0][0]])
    
    return dilations



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
        
    #dilations[0] = tf.constant(1)
    
    logits = drnn_classification(x,
                                 hidden_structs,
                                 dilations,
                                 n_classes,
                                 input_dims,
                                 cell_type)
    
    return (logits, params, dilations)




def dRNN_1(cell, dilated_inputs, rate, scope='default'):
    # building a dialated RNN with reformated (dilated) inputs
    dilated_outputs, _ = tf.contrib.rnn.static_rnn(
        cell, dilated_inputs,
        dtype=tf.float32, scope=scope)
    
    return dilated_outputs


def multi_dRNN(cells, inputs, dilations):
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
    x = inputs
    for cell in zip(cells, dilations):
        scope_name = "multi_dRNN_dilation_%d" % dilation
        x = dRNN(cell, x, scope=scope_name)
    return x


def reshape_inputs(inputs, rate):
    
    n_steps = len(inputs)
    if rate < 0 or rate >= n_steps:
        raise ValueError('The \'rate\' variable needs to be adjusted.')
    print "Building layer: %s, input length: %d, dilation rate: %d, input dim: %d." % (
        scope, n_steps, rate, inputs[0].get_shape()[1])

    # make the length of inputs divide 'rate', by using zero-padding
    EVEN = (n_steps % rate) == 0
    if not EVEN:
        # Create a tensor in shape (batch_size, input_dims), which all elements are zero.  
        # This is used for zero padding
        zero_tensor = tf.zeros_like(inputs[0])
        dialated_n_steps = n_steps // rate + 1
        print "=====> %d time points need to be padded. " % (
            dialated_n_steps * rate - n_steps)
        print "=====> Input length for sub-RNN: %d" % (dialated_n_steps)
        for i_pad in tf.range(dialated_n_steps * rate - n_steps):
            inputs.append(zero_tensor)
    else:
        dialated_n_steps = n_steps // rate
        print "=====> Input length for sub-RNN: %d" % (dialated_n_steps)

    # now the length of 'inputs' divide rate
    # reshape it in the format of a list of tensors
    # the length of the list is 'dialated_n_steps' 
    # the shape of each tensor is [batch_size * rate, input_dims] 
    # by stacking tensors that "colored" the same

    # Example: 
    # n_steps is 5, rate is 2, inputs = [x1, x2, x3, x4, x5]
    # zero-padding --> [x1, x2, x3, x4, x5, 0]
    # we want to have --> [[x1; x2], [x3; x4], [x_5; 0]]
    # which the length is the ceiling of n_steps/rate
    dilated_inputs = [tf.concat(inputs[i*rate : (i+1)*rate], axis=0) 
                      for i in range(dialated_n_steps)]
    
    return dilated_inputs



def reshape_outputs(dilated_outputs, rate):
    
    # reshape output back to the input format as a list of tensors with shape [batch_size, input_dims]
    # split each element of the outputs from size [batch_size*rate, input_dims] to 
    # [[batch_size, input_dims], [batch_size, input_dims], ...] with length = rate
    splitted_outputs = [tf.split(output, rate, axis=0)
                        for output in dilated_outputs]
    unrolled_outputs = [output
                        for sublist in splitted_outputs for output in sublist]
    # remove padded zeros
    outputs = unrolled_outputs[:n_steps]
    
    return outputs



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



    
    
    
    