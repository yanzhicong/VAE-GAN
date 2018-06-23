


import tensorflow as tf
import tensorflow.contrib.layers as tcl




def get_weightsinit(name, params=None):
    if name == 'normal' : 
        return tf.random_normal_initializer(float(params.split()[0]), float(params.split()[1]))
    elif name == 'xvarial' : 
        return None
    else :
        raise Exception("None weights initializer named " + name)

