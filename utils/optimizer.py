# -*- coding: utf-8 -*-
# MIT License
# 
# Copyright (c) 2018 ZhicongYan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

import tensorflow as tf
import tensorflow.contrib.layers as tcl


from .learning_rate import get_global_step
from .learning_rate import get_learning_rate


def get_optimizer(name, params, target, variables):
    if name == 'rmsprop':       
        return tf.train.RMSPropOptimizer(**params).minimize(target, var_list=variables)
    elif name == 'sgd':
        return tf.train.GradientDescentOptimizer(**params).minimize(target, var_list=variables)
    elif name == 'adam':
        return tf.train.AdamOptimizer(**params).minimize(target, var_list=variables)
    elif name == 'adadelta':
        return tf.train.AdadeltaOptimizer(**params).minimize(target, var_list=variables)
    else:
        raise Exception("None optimizer named " + name)


def get_optimizer_by_config(name, config, target, variables,    
                        global_step=None, 
                        global_step_update=None, 
                        global_step_name='global_step'):

    if global_step is None:
        global_step, global_step_update = get_global_step(global_step_name)

    if name == 'rmsprop':
        learning_rate = get_learning_rate(
                config.get('lr scheme', 'constant'),
                config['lr'],
                global_step,
                config.get('lr params', {})
            )

        optimize_op = tf.train.RMSPropOptimizer(
                learning_rate,
                decay=config.get('decay', 0.9),
                momentum=config.get('momentum', 0.0),
                epsilon=config.get('epsilon', 1e-10),
                centered=config.get('centered', False)
            ).minimize(target, var_list=variables)

    elif name == 'adam':
        if 'lr' in config:
            learning_rate = get_learning_rate(
                    config.get('lr scheme', 'constant'),
                    config['lr'],
                    global_step,
                    config.get('lr params', {})
                )

        else:
            learning_rate = tf.constant(0.001)
        optimize_op = tf.train.AdamOptimizer(
                learning_rate,
                beta1=config.get('beta1', 0.9),
                beta2=config.get('beta2', 0.999),
                epsilon=config.get('epsilon', 1e-8),
            ).minimize(target, var_list=variables)

    elif name == 'adadelta':
        if 'lr' in config:
            learning_rate = get_learning_rate(
                    config.get('lr scheme', 'constant'),
                    config['lr'],
                    global_step,
                    config.get('lr params', {})
                )

        else:
            learning_rate = tf.constant(0.001)
        optimize_op = tf.train.AdadeltaOptimizer(
                learning_rate,
                rho=config.get('rho', 0.95),
                epsilon=config.get('epsilon', 1e-8)
            ).minimize(target, var_list=variables)

    elif name == 'momentum': 
        learning_rate = get_learning_rate(
                config.get('lr scheme', 'constant'),
                config['lr'],
                global_step,
                config.get('lr params', {})
            )

        optimize_op = tf.train.MomentumOptimizer(
                learning_rate,
                config['momentum'],
                use_nesterov=config.get('use_nesterov', False)
            ).minimize(target, var_list=variables)

    elif name == 'sgd':
        learning_rate = get_learning_rate(
                config.get('lr scheme', 'constant'),
                config['lr'],
                global_step,
                config.get('lr params', {})
            )

        optimize_op = tf.train.GradientDescentOptimizer(
                learning_rate
            ).minimize(target, var_list=variables)



    if global_step_update is not None:
        return tf.group([optimize_op, global_step_update]), learning_rate, global_step
    else:
        return optimize_op, learning_rate, global_step
