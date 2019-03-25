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
    - input_: input spectrogram tensor
    - is_training: tf.bool placeholder
    '''
    def __init__(self, config, input_, is_training):
        self.config = config
        self.is_training = is_training
        onset_output, frame_output = self.build_model(input_)
        self.onset_output = onset_output
        self.frame_output = frame_output

    def build_model(self, input_)
        '''
        builds the frame prediction model
        '''
        onset_output = self._onset_model(input_)
        return onset_output, frame_output

    def _onset_model(self, input_):
        '''
        builds the neural network model for predicting the onsets
        input: mel spectrogram
        output: tensor of length 88, representing onset probabilities for each probability
        '''
        conv_output = self._conv_model(input_)
        return output

    def _conv_model(input_):
        '''
        builds the so-called acoustic network model
        input: mel spectrogram
        output: tensor of length 88
        '''
        return output


