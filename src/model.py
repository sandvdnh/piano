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
        self.reset_state = reset_state
        batch_size = config['batch_size']
        units = self.config['lstm_units']
        #self.input = input_


        self.c_fw = tf.zeros([batch_size, units])
        self.h_fw = tf.zeros([batch_size, units])
        self.c_bw = tf.zeros([batch_size, units])
        self.h_bw = tf.zeros([batch_size, units])
        self.c_fw_frames = tf.zeros([batch_size, units])
        self.h_fw_frames = tf.zeros([batch_size, units])
        self.c_bw_frames = tf.zeros([batch_size, units])
        self.h_bw_frames = tf.zeros([batch_size, units])


        onset_output, frame_output = self.build_model(input_)
        self.onset_output = onset_output
        self.frame_output = frame_output

    def build_model(self, input_):
        '''
        builds the frame prediction model
        '''
        units = self.config['lstm_units']
        batch_size = self.config['batch_size']
        sequence_length = self.config['sequence_length']
        onset_output = self._onset_model(input_)
        x = self._conv_model(input_, variable_scope='frame')

        #FRAMES
        x = tf.contrib.layers.fully_connected(
                x,
                88,
                activation_fn=tf.nn.sigmoid,
                weights_initializer=tf.initializers.truncated_normal)

        lstm_fw = tf.keras.layers.LSTM(
                units=units,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                bias_initializer=tf.zeros_initializer(),
                return_state=True,
                go_backwards=False,
                return_sequences=True)
        lstm_bw = tf.keras.layers.LSTM(
                units=units,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                bias_initializer=tf.zeros_initializer(),
                return_state=True,
                go_backwards=True,
                return_sequences=True)

        # update states depending on value of self.reset_state
        # FRAMES
        self.c_fw_frames = tf.cond(
                self.reset_state,
                true_fn=lambda: tf.zeros([batch_size, units]),
                false_fn=lambda: self.c_fw_frames)
        self.h_fw_frames = tf.cond(
                self.reset_state,
                true_fn=lambda: tf.zeros([batch_size, units]),
                false_fn=lambda: self.h_fw_frames)
        with tf.control_dependencies([self.c_fw_frames, self.h_fw_frames]):
            outputs_fw, self.h_fw_frames, self.c_fw_frames = lstm_fw(
                    x,
                    initial_state=(self.h_fw_frames, self.c_fw_frames),
                    training=self.is_training)

        # FRAMES
        # update states depending on value of self.reset_state
        self.c_bw_frames = tf.cond(
                self.reset_state,
                true_fn=lambda: tf.zeros([batch_size, units]),
                false_fn=lambda: self.c_bw_frames)
        self.h_bw_frames = tf.cond(
                self.reset_state,
                true_fn=lambda: tf.zeros([batch_size, units]),
                false_fn=lambda: self.h_bw_frames)
        with tf.control_dependencies([self.c_bw_frames, self.h_bw_frames]):
            outputs_bw, self.h_bw_frames, self.c_bw_frames = lstm_bw(
                    x,
                    (self.h_bw_frames, self.c_bw_frames),
                    training=self.is_training)
        # FRAMES
        output = tf.concat([outputs_fw, outputs_bw], axis=2)
        frame_output = tf.concat([output, onset_output], axis=2)
        frame_output = tf.contrib.layers.fully_connected(
                frame_output,
                88,
                activation_fn=tf.nn.sigmoid,
                weights_initializer=tf.initializers.truncated_normal)
        return onset_output, frame_output

    def _onset_model(self, input_):
        '''
        builds the neural network model for predicting the onsets
        INPUT: 
        - input_: shape (batch_size, sequence_size, 88)

        OUTPUT: tensor of length 88, representing onset probabilities for each probability

        '''
        x = self._conv_model(input_, variable_scope='onset')
        units = self.config['lstm_units']
        batch_size = self.config['batch_size']
        sequence_length = self.config['sequence_length']

        lstm_fw = tf.keras.layers.LSTM(
                units=units,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                bias_initializer=tf.zeros_initializer(),
                return_state=True,
                go_backwards=False,
                return_sequences=True)
        lstm_bw = tf.keras.layers.LSTM(
                units=units,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                bias_initializer=tf.zeros_initializer(),
                return_state=True,
                go_backwards=True,
                return_sequences=True)

        # update states depending on value of self.reset_state
        self.c_fw = tf.cond(
                self.reset_state,
                true_fn=lambda: tf.zeros([batch_size, units]),
                false_fn=lambda: self.c_fw)
        self.h_fw = tf.cond(
                self.reset_state,
                true_fn=lambda: tf.zeros([batch_size, units]),
                false_fn=lambda: self.h_fw)
        with tf.control_dependencies([self.c_fw, self.h_fw]):
            outputs_fw, self.h_fw, self.c_fw = lstm_fw(
                    x,
                    initial_state=(self.h_fw, self.c_fw),
                    training=self.is_training)

        # update states depending on value of self.reset_state
        self.c_bw = tf.cond(
                self.reset_state,
                true_fn=lambda: tf.zeros([batch_size, units]),
                false_fn=lambda: self.c_bw)
        self.h_bw = tf.cond(
                self.reset_state,
                true_fn=lambda: tf.zeros([batch_size, units]),
                false_fn=lambda: self.h_bw)
        with tf.control_dependencies([self.c_bw, self.h_bw]):
            outputs_bw, self.h_bw, self.c_bw = lstm_bw(
                    x,
                    (self.h_bw, self.c_bw),
                    training=self.is_training)

        output = tf.concat([outputs_fw, outputs_bw], axis=2)

        #shape = tf.TensorShape([batch_size * sequence_length, 2 * units])
        #output = tf.reshape(output, shape=shape)
        output = tf.contrib.layers.fully_connected(
                output,
                88,
                activation_fn=tf.nn.sigmoid,
                weights_initializer=tf.initializers.truncated_normal)
        #output = tf.divide(output, tf.expand_dims(tf.reduce_sum(output, axis=2), axis=2))
        return output

    def _conv_model(self, input_, variable_scope):
        '''
        builds the so-called acoustic network model
        INPUT:
        - mel spectrogram (batch_size, 5, n_freq_bins=229, 1)

        OUTPUT:
        - x: tensor of shape (batch_size, sequence_length, 88)
        '''
        with tf.variable_scope(variable_scope):
            x = input_
            shape = tf.TensorShape(
                    [
                        self.config['batch_size'] * self.config['sequence_length'],
                        5,
                        self.config['spec_n_bins'],
                        1
                    ])
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


