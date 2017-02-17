# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 01:02:09 2016

Some operation methods are from DCGAN tensorflow github: https://github.com/carpedm20/DCGAN-tensorflow
@author: yuhuachen
"""
import tensorflow as tf;
import numpy as np
import scipy.misc
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm


# Utils
def imread(path, c_dim=3):
    """
    Read images
    :param path: input image file path & name
    :param c_dim: color dimensions, default = 3
    :return: numpy images in uint8 format
    """
    im = scipy.misc.imread(path).astype(np.float)
    if c_dim == 1:
        im = np.asarray(np.dstack((im, im, im)), dtype=np.uint8)
    return im;


def fc2d(input__, output_dim, name='fc2d', stddev=0.02, reuse=False):
    """
    Fully connected layers for 2d Image inputs
    :param input__:     4d input tensor (batch, height, width, channels)
    :param output_dim:  out_put channel dimensions
    :param name:        name of the layer/operation
    :param stddev:      standard deviation for initializer
    :param reuse:       reuse for variable field
    :return:            layer output, weights, biases
    """
    with tf.variable_scope(name, reuse=reuse):
        in_shapes = [int(_) for _ in input__.get_shape()];
        input__ = tf.reshape(input__, [-1,
                                       in_shapes[1] * in_shapes[2] * in_shapes[-1]]);
        w = tf.get_variable('w', [in_shapes[1] * in_shapes[2] * in_shapes[-1],
                                  output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev)
                            );

        b = tf.get_variable('b', [output_dim],
                            initializer=tf.constant_initializer(0.0)
                            );

        fc = tf.matmul(input__, w) + b;

        return fc, w, b;


def fc1d(input__, output_dim, name='fc1d', stddev=0.02, reuse=False):
    """
    Fully connected layers for 1d inputs
    :param input__:     4d input tensor (batch, length, channels)
    :param output_dim:  out_put channel dimensions
    :param name:        name of the layer/operation
    :param stddev:      standard deviation for initializer
    :param reuse:       reuse for variable field
    :return:            layer output, weights, bias
    """
    with tf.variable_scope(name, reuse=reuse):
        in_shapes = [int(_) for _ in input__.get_shape()];
        w = tf.get_variable('w', [in_shapes[1],
                                  output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev)
                            );
        b = tf.get_variable('b', [output_dim],
                            initializer=tf.constant_initializer(0.0)
                            );

        fc = tf.matmul(input__, w) + b;

        return fc, w, b;


def conv2d(input__, output_dim, k_h=5, k_w=5, d_h=1, d_w=1,
           stddev=0.02, data_format='NHWC', name='conv2d', padding='SAME', reuse=False):
    """
    2D Convolution layer
    :param input__:     4d input tensor (batch, height, width, channels)
    :param output_dim:  filter number
    :param k_h:         kernel height
    :param k_w:         kernel width
    :param d_h:         stride height
    :param d_w:         stride width
    :param stddev:      standard deviation for initializer
    :param data_format: data format, default is (batch, height, width, channels)
    :param name:        name of the layer
    :param padding:     padding mode
    :param reuse:       reuse for variable field
    :return:            layer output, weights, bias
    """
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable('w', [k_h, k_w, input__.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        biases = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(input__, w, strides=[1, d_h, d_w, 1],
                            padding=padding, data_format=data_format)

        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv, w, biases


def batch_norm_layer(x, train_phase, name='batch_norm', reuse=False):
    """
    Batch normalization layer
    :param x:               Input tensor
    :param train_phase:     Boolean variable to determine whether it is in training phase
    :param name:            name of the layer
    :param reuse:           reuse for variable field
    :return:                layer output
    """
    with tf.variable_scope(name, reuse=reuse):
        bn_train = batch_norm(x, decay=0.99, center=True, scale=True,
                              is_training=True,
                              updates_collections=None,  # TODO: make it to a udpate collection
                              reuse=False,  # is this right?
                              trainable=True,
                              scope=tf.get_variable_scope())
        bn_inference = batch_norm(x, decay=0.99, center=True, scale=True,
                                  is_training=False,
                                  updates_collections=None,  # TODO: make it to a udpate collection
                                  reuse=True,  # is this right?
                                  trainable=True,
                                  scope=tf.get_variable_scope())
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z


def pool2d(input__, k_h=5, k_w=5, d_h=5, d_w=5, padding='SAME',
           data_format='NHWC', name='pool2d'):
    """
    2D pooling layer
    :param input__:     4d input tensor (batch, height, width, channels)
    :param k_h:         kernel height
    :param k_w:         kernel width
    :param d_h:         stride height
    :param d_w:         stride width
    :param padding:     padding mode
    :param data_format: data format, default is (batch, height, width, channels)
    :param name:        name of the layer
    :return:            layer output
    """
    _pooled = tf.nn.max_pool(input__, [1, k_h, k_w, 1], [1, d_h, d_w, 1],
                             padding=padding, name=name, data_format=data_format);
    return _pooled;


def deconv2d(input__, output_dim, batch_size, output_size, k_h=5, k_w=5, d_h=1, d_w=1,
             stddev=0.02, data_format='NHWC', name='deconv2d', padding='SAME', reuse=False):

    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable('w', [k_h, k_w, output_dim, input__.get_shape()[-1]],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        biases = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))

        output_shape = [tf.cast(batch_size, dtype=tf.int32).eval(),
                        tf.cast(output_size, dtype=tf.int32).eval(),
                        tf.cast(output_size, dtype=tf.int32).eval(),
                        tf.cast(output_dim, dtype=tf.int32).eval()];

        conv = tf.nn.conv2d_transpose(input__, w, output_shape, strides=[1, d_h, d_w, 1],
                                      data_format=data_format, padding=padding)

        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv, w, biases


def lrelu(x, leak=0.2, name="lrelu"):
    """
    Leaky relu
    :param x:       Input
    :param leak:    leak of the relu
    :param name:    name of the layer
    :return:        activation output
    """
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)
