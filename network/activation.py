

import tensorflow as tf
import tensorflow.contrib.layers as tcl



def get_lrelu(params):
    if params == None:
        leak = 0.1
    else :
        leak = float(params.split()[0])

    def lrelu(x, leak=leak, name="lrelu"):
        with tf.variable_scope(name):
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)
    return lrelu

# activation_dict = {
#     "relu" : tf.nn.relu,
#     "lrelu" : lrelu,
#     "softmax" : tf.nn.softmax,
#     "sigmoid" : tf.nn.sigmoid
# }


def get_activation(name, params=None):
    # if name in activation_dict : 
        # return activation_dict[name]

    if name == 'relu':
        return tf.nn.relu
    elif name == 'lrelu':
        return get_lrelu(params)
    elif name == 'softmax' : 
        return tf.nn.softmax
    elif name == 'sigmoid':
        return tf.nn.sigmoid
    else :
        raise Exception("None actiavtion named " + name)




