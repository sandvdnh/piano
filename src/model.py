from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import glob

class Trainer():
    def __init__():
        pass

    def build_trainer():
        pass


class Model():
    '''
    model class for the neural network
    input:
    - config: config file
    - input_: input spectrogram tensor (batch_size, 5, 229, 1)
    - is_training: tf.bool placeholder
    '''
    def __init__(self, config, input_, is_training):
        self.config = config
        self.is_training = is_training
        onset_output, frame_output = self.build_model(input_)
        self.onset_output = onset_output
        self.frame_output = frame_output

    def build_model(self, input_):
        '''
        builds the frame prediction model
        '''
        onset_output = self._onset_model(input_)
        frame_output = tf.constant(1, dtype=tf.int64)
        return onset_output, frame_output

    def _onset_model(self, input_):
        '''
        builds the neural network model for predicting the onsets
        input: mel spectrogram
        output: tensor of length 88, representing onset probabilities for each probability
        '''
        conv_output = self._conv_model(input_)
        return conv_output

    def _conv_model(self, input_):
        '''
        builds the so-called acoustic network model
        input: mel spectrogram (batch_size, 5, n_freq_bins=229, 1)
        output: tensor of length 88
        '''
        x = input_
        shape = tf.TensorShape([self.config['batch_size'], 5, self.config['spec_n_bins'], 1])
        x = tf.reshape(x, shape=shape)
        filter1 = tf.get_variable(
                'filter1',
                dtype=tf.float32,
                shape=[3, 3, 1, 32],
                initializer=tf.initializers.truncated_normal,
                trainable=True)
        x = tf.nn.conv2d(
                x,
                filter1,
                strides=[1, 1, 1, 1],
                padding='SAME')
        x = tf.contrib.layers.batch_norm(x, is_training=self.is_training)
        x = tf.nn.relu(x)

        filter2 = tf.get_variable(
                'filter2',
                dtype=tf.float32,
                shape=[3, 3, 32, 32],
                initializer=tf.initializers.truncated_normal,
                trainable=True)
        x = tf.nn.conv2d(
                x,
                filter2,
                strides=[1, 1, 1, 1],
                padding='SAME')
        x = tf.contrib.layers.batch_norm(x, is_training=self.is_training)
        x = tf.nn.relu(x)
        x = tf.contrib.layers.max_pool2d(
                x,
                kernel_size=(2, 2),
                stride=2,
                padding='VALID')
        x = tf.nn.dropout(
                x,
                keep_prob=0.25)

        filter3 = tf.get_variable(
                'filter3',
                dtype=tf.float32,
                shape=[3, 3, 32, 64],
                initializer=tf.initializers.truncated_normal,
                trainable=True)
        x = tf.nn.conv2d(
                x,
                filter3,
                strides=[1, 1, 1, 1],
                padding='SAME')
        x = tf.contrib.layers.max_pool2d(
                x,
                kernel_size=(2, 2),
                stride=2,
                padding='VALID')
        x = tf.nn.dropout(
                x,
                keep_prob=0.25)

        x_shape = tf.shape(x)
        x = tf.reshape(x, shape=[x_shape[0], x_shape[1] * x_shape[2] * x_shape[3]])

        x = tf.contrib.layers.fully_connected(
                x,
                512,
                activation_fn=None,
                weights_initializer=tf.initializers.truncated_normal)
        x = tf.nn.dropout(
                x,
                keep_prob=0.5)
        x = tf.contrib.layers.fully_connected(
                x,
                88,
                activation_fn=None,
                weights_initializer=tf.initializers.truncated_normal)
        return x


