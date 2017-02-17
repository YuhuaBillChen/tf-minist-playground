"""
Tensorflow model for training MNIST dataset
"""
import tensorflow as tf
import numpy as np
import os
import sropts
from neural_networks import *


class MNISTModel(object):
    class BaseModel(object):
        """
        Base model
        """

        def __init__(self, data_dir="./data/MNIST", chkpt_dir="./chkpt/", batch_size=64, image_size=28, category=10,
                     learning_rate=1e-3):
            """
            Initialization of base model
            :param data_dir:        Data cache folder for MNSIT data
            :param chkpt_dir:       Checkpoints directory
            :param batch_size:      Batch size
            :param image_size:      Image size
            :param category:        Category number
            :param learning_rate:   Learning rate
            """
            # Parameter settings
            self.data_dir = os.path.realpath(data_dir);
            self.chkpt_dir = os.path.realpath(chkpt_dir);
            self.batch_size = batch_size;
            self.image_size = image_size;
            self.category = category;
            self.learning_rate = learning_rate;

            # Initializations
            # Graph operations
            self.predict_op = None;
            self.m_loss_op = None;
            self.accuracy_op = None;
            self.opt_op = None;
            self.init_op = None;
            self.saver = None;

            self.mnist = self.get_data();  # Data object
            self.x, self.y, self.is_train = self.init_ph();  # Place hodlers


            # Create sub folders
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir);
            if not os.path.exists(self.chkpt_dir):
                os.makedirs(self.chkpt_dir);
            return;

        def get_data(self):
            """
            Get/Download MNIST data
            :return:
            """
            from tensorflow.examples.tutorials.mnist import input_data
            return input_data.read_data_sets(self.data_dir, one_hot=True);

        def init_ph(self):
            """
            Initialize place holders
            :return:
            """
            x = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.image_size, self.image_size, 1),
                               name='in-img');
            y = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.category), name='in-label');
            is_train = tf.placeholder(dtype=tf.bool, shape=[], name='is_train');
            return x, y, is_train;

        def visualized(self):
            return;

        def build(self, h_dim=32, fc_dim=256, block_num=8):
            """
            Build the network with settings
            :param h_dim:
            :param fc_dim:
            :param block_num:
            :return:
            """
            m_nn = NeuralNetworks.ConvMNIST(h_dim=32, fc_dim=1024, block_num=5, is_train=self.is_train);
            self.predict_op = m_nn.nn_predict(self.x, reuse=False);
            self.m_loss_op = m_nn.loss(predict=self.predict_op, real=self.y);
            correct_prediction = tf.equal(tf.argmax(self.predict_op, 1), tf.argmax(self.y, 1))
            self.accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            self.opt_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.m_loss_op);
            self.init_op = tf.global_variables_initializer();

            self.saver = tf.train.Saver();

        def train(self, sess, epoch_num=25):
            """
            Training phase of the model
            :param sess:        tensorflow session instance
            :param epoch_num:   epochs to train
            :return:
            """
            if self.predict_op is None:
                self.build();
            sess.run(self.init_op);
            best_val_acc = 0;
            for epoch in xrange(epoch_num):
                count = 0;
                total_acc = []
                while count < self.mnist.train.num_examples:
                    count += self.batch_size;
                    [input_x, input_y] = self.mnist.train.next_batch(batch_size=self.batch_size);
                    input_x = np.reshape(input_x, newshape=(self.batch_size, 28, 28, 1));

                    [comp_loss, comp_acc, __] = sess.run([self.m_loss_op, self.accuracy_op, self.opt_op],
                                                         feed_dict={
                                                             self.x: input_x,
                                                             self.y: input_y,
                                                             self.is_train: True
                                                         });
                    total_acc.append(comp_acc);
                    if count / self.batch_size % 10 == 0:
                        print "\r\bepoch:", epoch, " image#:", count, " avg acc:", np.mean(
                            total_acc), " loss:", comp_loss, "acc:", comp_acc,

                # Valuation
                pred_labels, real_labels = self.predict(sess, num_of_images=self.mnist.validation.num_examples,
                                                        dataset=self.mnist.validation);

                val_corr = np.equal(np.argmax(pred_labels, 1), np.argmax(real_labels, 1));
                val_acc = np.mean(val_corr.astype(np.float32));
                print "val: ", val_acc,
                if val_acc > best_val_acc:
                    self.saver.save(sess, self.chkpt_dir + "model.chkpt")
                    best_val_acc = val_acc;
                    print "model saved!";
                else:
                    print "";

        def test(self, sess):
            """
            Test phase
            :param sess:    Tensorflow session instance
            :return:
            """
            if self.predict_op is None:
                self.build();
            self.saver.restore(sess, self.chkpt_dir + "model.chkpt");
            pred_labels, real_labels = self.predict(sess, num_of_images=self.mnist.test.num_examples,
                                                    dataset=self.mnist.test);
            print "Forward completed";
            test_corr = np.equal(np.argmax(pred_labels, 1), np.argmax(real_labels, 1));
            test_acc = np.mean(test_corr.astype(np.float32));
            print "average acc:", test_acc;
            return pred_labels, real_labels;


        def predict(self, sess, num_of_images, dataset):
            """
            Make a prediction for a given number of images
            :param sess:                tensorflow session instance
            :param num_of_images:       number of images to predict
            :param dataset:             mnist.train or mnist.test
            :return: predicted labels, real labels
            """
            real_labels = [];
            pred_labels = [];
            count = 0;
            while count < num_of_images:
                count += self.batch_size;
                [input_x, input_y] = dataset.next_batch(batch_size=self.batch_size);
                input_x = np.reshape(input_x, newshape=(self.batch_size, self.image_size, self.image_size, 1));
                [comp_pred] = sess.run([self.predict_op], feed_dict={
                    self.x: input_x,
                    self.y: input_y,
                    self.is_train: False
                });
                pred_labels.extend(comp_pred);
                real_labels.extend(input_y);

            return pred_labels, real_labels;
