# -*- coding: utf-8 -*-
"""
Neural Network class 
Created on Wed Dec 21 09:52:04 2016

@author: bill
"""
import tensorflow as tf;

from sropts import *


class NeuralNetworks(object):
    """
    Neural Network root class
    Defines common interface for different networks:
        nn_predict: to make a prediction
        nn_build:
    """

    class NeuralNetwork(object):
        def __init__(self):
            """
            Default initialization of the network class
            """
            self.loss_func = 0;
            self.is_train = None;
            return;

        def nn_predict(self, input__, reuse=False):
            """
                This method is to make prediction from the inputs
                :param input__    a 4D Tensor
                :param reuse      If the variable in the network is designed to be reusable
            """
            return None;

        def nn_build(self, inputs, reuse=False):
            """
                This method is to generate the image as well as compute the loss function
                :param inputs    a list 4D Tensor. (Typical use, 1st 4D Tensor is LR images, 2nd is HR images)
                :param reuse     If the variable in the network is designed to be reusable
                :return
            """
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
        def __init__(self, h_dim, fc_dim, is_train, dropout=0.5, block_num=4, loss_func=0):
            """
            Convolutional neural networks for MNIST task
            Structure: 4 convolution layers, doubling the filter number
            :param h_dim:       hidden dimension for the first convolution layer
            :param fc_dim:      Fully connected layer dimension
            :param is_train     Bool variable to indicate whether it is in training phase
            :param dropout:     keep rate of dropout layer for fully connect layers
            :param block_num    How many residual network blocks
            :param loss_func:   which loss_func to pick
            """
            self.h_dim = h_dim;
            self.fc_dim = fc_dim;
            self.loss_func = loss_func;
            self.dropout = dropout;
            self.is_train = is_train;
            self.block_num = block_num;
            return;

        def nn_block(self, input__, h_dim, reuse, block_id):
            # First Conv layer
            [h0, _, _] = conv2d(input__, h_dim, k_h=3, k_w=3, d_h=1, d_w=1, name='conv%d-1' % block_id, reuse=reuse);
            act_h0 = lrelu(h0, name='lrelu%d-0' % block_id);

            # Second Layer
            [h1, _, _] = conv2d(act_h0, h_dim, k_h=3, k_w=3, d_h=1, d_w=1, name='conv%d-2' % block_id,
                                reuse=reuse);
            act_h1 = lrelu(h1 + h0, name='lrelu%d-1' % block_id);

            # Pooling and batch normal
            pool_h1 = pool2d(act_h1, d_h=2, d_w=2);

            return batch_norm_layer(pool_h1, train_phase=self.is_train, name="bn%d-1" % block_id, reuse=reuse);

        def nn_predict(self, input__, reuse=False):

            h_dim = self.h_dim;  # Intermediate filter number
            inter_x = input__;  # Intermediate data
            for block_id in xrange(self.block_num):
                inter_x = self.nn_block(inter_x, h_dim, reuse, block_id);
                if block_id > 1:
                    h_dim = 2 * self.h_dim;  # Twice the filter number

            # Fully connected layer
            fc1, _, _ = fc2d(inter_x, self.fc_dim, name='fc1', reuse=reuse);

            # Drop out #1
            fc1_drop = tf.cond(self.is_train, lambda: tf.nn.dropout(fc1, self.dropout), lambda: tf.nn.dropout(fc1, 1.0));

            # Final read out layer
            [predict, _, _] = fc1d(fc1_drop, 10, name='fc2', reuse=reuse);

            return predict;
