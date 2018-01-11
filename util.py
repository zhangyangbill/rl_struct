import numpy as np
import tensorflow as tf
from collections import OrderedDict

def xor_data(num_obs, size):
    '''This function generates toy xor data
    
    Args:
    num_obs - number of tokens to be generated
    size - an array of size of the data
    
    Returns:
    inputs - a [num_obs, size] array of Bernoulli random variables taking 1 and -1.
    target - a length num_obs array of the occurence of the minority sign.
    '''
    
    inputs = np.random.uniform(size = [num_obs] + size) * 2.0 - 1.0
    targets = np.sum(np.reshape(inputs>0, [num_obs, -1]), axis = 1)
    targets = np.minimum(targets, np.prod(size) - targets)
    
    return inputs, targets

def LBFGS(step_delta, grad_delta, x):
    ''' This function performs L-BFGS to compute H^{-1} x, where H is a Hessian matrix.
    See http://aria42.com/blog/2014/12/understanding-lbfgs (typos, inaccurate)
        https://en.wikipedia.org/wiki/Limited-memory_BFGS
    
    Args:
    step_delta - a list of differences in function argument. Each element can be a list of numpy arrays
    grad_delta - a list of differences in gradient. Each element can be a list of numpy arrays
    x - The vector the inverse Hessian is multiplied with. Should have the same dimension as each element in step_delta and grad_delta.
    
    Returns:
    H^{-1} x. Should have the same dimension as each element in step_delta and grad_delta
    '''

    # number of step_delta, grad_delta pairs
    num_deltas = len(step_delta)
    
    # initialize
    r = x
    rho = [0] * num_deltas
    alpha = [0] * num_deltas
    
    # compute right product
    for it in reversed(xrange(num_deltas)):
        rho[it] = 1 / sum([np.sum(yy * ss) for yy, ss in zip(grad_delta[it], step_delta[it])])
        alpha[it] = sum([rho[it] * np.sum(ss * rr) for ss, rr in zip(step_delta[it], r)])
        r = [rr - alpha[it] * yy for rr, yy in zip(r, grad_delta[it])]
    
    # scale
    gamma = 1 / sum([np.sum(yy * yy) for yy in grad_delta[num_deltas-1]]) / rho[num_deltas-1]
    r = [rr * gamma for rr in r]
    
    # compute left product
    for it in xrange(num_deltas):
        beta = sum([rho[it] * np.sum(yy * rr) for yy, rr in zip(grad_delta[it], r)])
        r = [rr + (alpha[it] - beta) * ss 
             for rr, ss in zip(r, step_delta[it])]
        
    return r



def extract_update_dict(update_ops):
    """Extract variables and their new values from Assign and AssignAdd ops.
    
    Args:
        update_ops: list of Assign and AssignAdd ops, typically computed using Keras' opt.get_updates()

    Returns:
        dict mapping from variable values to their updated value
    """
    name_to_var = {v.name: v for v in tf.global_variables()}
    updates = OrderedDict()
    for update in update_ops:
        var_name = update.op.inputs[0].name
        var = name_to_var[var_name]
        value = update.op.inputs[1]
        if update.op.type == 'Assign':
            updates[var.value()] = value
        elif update.op.type == 'AssignAdd':
            updates[var.value()] = var + value
        else:
            raise ValueError("Update op type (%s) must be of type Assign or AssignAdd"%update_op.op.type)
    return updates

def remove_original_op_attributes(graph):
    """Remove _original_op attribute from all operations in a graph."""
    for op in graph.get_operations():
        op._original_op = None
        
def graph_replace(*args, **kwargs):
    """Monkey patch graph_replace so that it works with TF 1.0"""
    _graph_replace = tf.contrib.graph_editor.graph_replace
    remove_original_op_attributes(tf.get_default_graph())
    return _graph_replace(*args, **kwargs)
