"""
Customized data batching class based on tensorflow's MNIST helper module
"""
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np;

class MNISTData(object):
    class BaseImages(object):
        """
        Completely identical to mnist dataset
        """
        def __init__(self, data_dir):
            self.data_dir = data_dir;
            self.mnist = input_data.read_data_sets(self.data_dir, one_hot=True);
            self.train = self.mnist.train;
            self.validation = self.mnist.validation;
            self.test = self.mnist.test;

    class NoisyImages(BaseImages):
        """
        Add Gaussian Noise to Images
        """
        def __init__(self, data_dir, stddev=8):
            self.data_dir = data_dir;
            self.stddev = stddev;
            self.mnist = input_data.read_data_sets(self.data_dir, one_hot=True);
            self.train = MNISTData.NoisyImages.DataSet(self.mnist.train, stddev);
            self.validation = MNISTData.NoisyImages.DataSet(self.mnist.validation);
            self.test = self.mnist.test;                    # Test set is still ground truth

        class DataSet(object):
            """
            Helper class to wrap up mnist data sets
            """

            def __init__(self, data_set, stddev):
                self.data_set = data_set;
                self.stddev = stddev;
                self.images = None;      # Should not be called here
                self.labels = data_set.labels;
                self.num_examples = data_set.num_examples;

            def next_batch(self, batch_size):
                """
                Generate next batch of noisy images and labels
                :param batch_size: batch size of the images
                :return: images, labels

                """
                [input_x, input_y] = self.data_set.next_batch(batch_size);
                output_x = self.noisy_images(input_x);
                return [output_x, input_y];

            def noisy_images(self, input_x):
                output_x = np.zeros(input_x.shape);
                for i, im in enumerate(input_x):
                    noisy = np.array(np.clip(255 * im + self.stddev * np.random.randn(im.shape[0]), 0, 255), dtype=np.uint8);
                    output_x[i, :] = noisy;
                return np.array(output_x, dtype=np.float32)/255;

    class NoisyLabels(BaseImages):
        """
        Add Gaussian Noise to Images
        """
        def __init__(self, data_dir, percent=0.05):
            self.data_dir = data_dir;
            self.percent = percent;
            self.mnist = input_data.read_data_sets(self.data_dir, one_hot=True);
            self.train = MNISTData.NoisyLabels.DataSet(self.mnist.train, percent)       # Only the training is noisy
            self.validation = MNISTData.NoisyLabels.DataSet(self.mnist.validation, percent=percent);
            self.test = self.mnist.test;                # Test set is still ground truth

        class DataSet(object):
            """
            Helper class to wrap up mnist data sets
            """

            def __init__(self, data_set, percent):
                self.data_set = data_set;
                self.percent = percent;
                self.images = data_set.images;
                self.labels = None;      # Should not be called here
                self.num_examples = data_set.num_examples;

            def next_batch(self, batch_size):
                """
                Generate next batch of noisy images and labels
                :param batch_size: batch size of the images
                :return: images, labels

                """
                [input_x, input_y] = self.data_set.next_batch(batch_size);
                output_y = self.noisy_labels(input_y);
                return [input_x, output_y];

            def noisy_labels(self, input_y):
                output_y = np.array(input_y);
                for i in xrange(output_y.shape[0]):
                    if np.random.random_sample() < self.percent:
                        while np.alltrue(np.equal(output_y[i], input_y[i])):
                            np.random.shuffle(output_y[i]);
                return output_y;

