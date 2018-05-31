

import tensorflow as tf
import tensorflow.contrib.layers as tcl



def get_normalization(name, params):
    if name == 'batch_norm' : 
        return tcl.batch_norm, params
    elif name == 'none':
        return None, None
    else:
        raise Exception("None normalization named " + name)


