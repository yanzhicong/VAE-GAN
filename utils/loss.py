



import tensorflow as tf
import tensorflow.contrib.layers as tcl



def kl_loss(z_avg, z_log_var):
    kl_loss = -1.5 * tf.mean(1.0 + z_log_var - tf.exp(z_log_var) - tf.square(z_avg))





