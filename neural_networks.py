# -*- coding: utf-8 -*-
"""
Neural Network class 
Created on Wed Dec 21 09:52:04 2016

@author: bill
"""
import tensorflow as tf;

from sropts import *


class NeuralNetworks(object):
    class NeuralNetwork(object):
        def __init__(self):
            """
            loss_func:  the type of loss function, default is 0 for cross_entropy.
            is_train:   Whether it is in the training phase (Used in batch_normalization layer)
            """
            self.loss_func = 0;
            self.is_train = None;
            return;

        def nn_predict(self, input__, reuse=False):
            '''
                This method is to generate image from the input
                input__:    a 4D Tensor
                resuse:     If the variable in the network is designed to be reusable
            '''
            return None;

        def nn_build(self, inputs, reuse=False):
            '''
                This method is to generate the image as well as compute the loss function
                input__:    a list 4D Tensor. (Typical use, 1st 4D Tensor is LR images, 2nd is HR images)
                resuse:     If the variable in the network is designed to be reusable
            '''
            if len(inputs) == 0:
                return None;

            predict = self.nn_predict(inputs[0], reuse=reuse);

            # Compute loss function if needed
            if len(inputs) > 1:
                g_loss = tf.reduce_mean(self.loss(predict, inputs[1]));
            else:
                g_loss = None;
            return predict, g_loss;

        def loss(self, predict, real):
            """
            Compute the loss between predict and real.
            The exact type of loss function is determined by self.loss_func variable
            :param predict: predict values
            :param real:    real values
            :return:        loss between prediction and real values
            """
            if self.loss_func == 0:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict, real));

            return loss

    class ConvMNIST(NeuralNetwork):
        def __init__(self, h_dim, loss_func=0):
            """
            Convolutional neural networks for MNIST task
            :param h_dim:       hidden dimension for the first convolution layer
            :param loss_func:   which loss_func to pick
            """
            self.h_dim = h_dim;
            self.loss_func = loss_func;
            return;

        def nn_predict(self, input__, reuse=False):
            if input__.get_shape()[-1] == 1:
                color_channel = False;
            else:
                color_channel = True;
            # First Conv layer
            [self.h0, self.w0, self.b0] = conv2d(input__, self.h_dim, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1',
                                                 reuse=reuse);
            act_h0 = lrelu(self.h0, name='lrelu0')

            pool_h0 = pool2d(act_h0, d_h=2, d_w=2);

            # Second Layer
            [self.h1, self.w1, self.b1] = conv2d(pool_h0, self.h_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2',
                                                 reuse=reuse);
            act_h1 = lrelu(self.h1, name='lrelu1')

            pool_h1 = pool2d(act_h1, d_h=2, d_w=2);

            # Third Layer
            [self.h2, self.w2, self.b2] = conv2d(pool_h1, self.h_dim * 4, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3',
                                                 reuse=reuse);
            act_h2 = lrelu(self.h2, name='lrelu2')

            pool_h2 = pool2d(act_h2, d_h=2, d_w=2);


            # Fourth Layer
            [self.h3, self.w3, self.b3] = conv2d(pool_h2, self.h_dim * 4, k_h=3, k_w=3, d_h=1, d_w=1, name='conv4',
                                                 reuse=reuse);
            act_h3 = lrelu(self.h3, name='lrelu3')

            pool_h3 = pool2d(act_h3, d_h=2, d_w=2);

            # Fully connected layers
            [self.h4, self.h4_w, self.h4_b] = fc2d(pool_h3, self.h_dim, name='fc1', reuse=reuse);

            act_h4 = lrelu(self.h4, name='lrelu5');

            [self.h5, self.h5_w, self.h5_b] = fc1d(act_h4, 10, name='fc2', reuse=reuse);

            return self.h5;

