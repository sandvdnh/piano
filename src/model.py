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
    def __init__(self, config, input_, is_training, reset_state):
        self.config = config
        self.is_training = is_training
        sequence_length = self.config['sequence_length']
        units = self.config['lstm_units']
        self.c_fw = tf.zeros([units])
        self.h_fw = tf.zeros([units])
        self.c_bw = tf.zeros([units])
        self.h_bw = tf.zeros([units])
        self.reset_state = reset_state
        onset_output, frame_output = self.build_model(input_)
        self.onset_output = onset_output
        self.frame_output = frame_output
        self.onset_state = None # state of the LSTM in the onset detection network
        self.offset_state = None # state of the LSTM in the offset detection network

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
        INPUT: 
        - input_: shape (batch_size, sequence_size, 88)

        OUTPUT: tensor of length 88, representing onset probabilities for each probability

        The LSTM contains 128 units in both forward and backward directions
        '''
        x = self._conv_model(input_)
        # x has dimensions (batch_size, sequence_size, 88)

        units = self.config['lstm_units']
        batch_size = self.config['batch_size']
        sequence_length = self.config['sequence_length']

        lstm_fw = tf.keras.layers.CuDNNLSTM(
                units=units,
                kernel_initializer=
                tf.contrib.layers.variance_scaling_initializer(),
                bias_initializer=tf.zeros_initializer(),
                return_state=True,
                go_backwards=False,
                return_sequences=True)

        lstm_bw = tf.keras.layers.CuDNNLSTM(
                units=units,
                kernel_initializer=
                tf.contrib.layers.variance_scaling_initializer(),
                bias_initializer=tf.zeros_initializer(),
                return_state=True,
                go_backwards=True,
                return_sequences=True)

        if self.reset_state:
            self.c_fw = tf.zeros([1, sequence_length, units])
            self.h_fw = tf.zeros([1, sequence_length, units])
        outputs_fw, states = lstm_fw(
                x,
                (self.h_fw, self.c_fw),
                training=self.is_training)
        # update states
        self.h_fw = states[0]
        self.c_fw = states[1]
        if self.reset_state:
            self.c_bw = tf.zeros([1, sequence_length, units])
            self.h_bw = tf.zeros([1, sequence_length, units])
        outputs_bw, states = lstm_bw(
                x,
                (self.h_bw, self.c_bw),
                training=self.is_training)
        # update states
        self.h_bw = states[0]
        self.c_bw = states[1]

        return outputs_bw

    def _conv_model(self, input_):
        '''
        builds the so-called acoustic network model
        INPUT:
        - mel spectrogram (batch_size, 5, n_freq_bins=229, 1)

        OUTPUT:
        - x: tensor of shape (batch_size, sequence_length, 88)
        '''
        x = input_
        shape = tf.TensorShape([self.config['batch_size'] * self.config['sequence_length'], 5, self.config['spec_n_bins'], 1])
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

        # assert x_shape[1] == 1? 

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

        shape = tf.TensorShape([self.config['batch_size'], self.config['sequence_length'], 88])
        x = tf.reshape(x, shape=shape)
        return x


