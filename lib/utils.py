from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import glob
import os
import librosa
import soundfile as sf
from scipy.io.wavfile import read
import tensorflow as tf


def read_files(config, except_files):
    '''
    Reads in text files in config['path'] and returns an (n, 2) ndarray
    with intervals and an (n,) ndarray with pitches in Hz

    returns a list of tuples (a, b, c), where each tuple corresponds to a file:
    - a: cqt transform of .wav file
    - b: list of intervals from .txt file
    - c: corresponding list of pitches (in MIDI)
    '''
    print('READING FILES...')
    data_list = []
    dirs = config['dirs'] # list of names of subdirectories in MAPS dataset to be included
    file_list = []
    names = []
    loaded_files = []
    for _ in dirs:
        path = os.path.join(config['path'], _, 'MUS')
        path = os.path.join(path, '*.wav')
        wav_files = glob.glob(path)
        for wav_file in wav_files:
            name, _ = os.path.splitext(wav_file)
            names.append(name)
            txt_file = name + '.txt'
            file_list.append((wav_file, txt_file))
        count = 0
        for i, pair in enumerate(file_list):
            if count < config['files_to_load'] and pair[0] not in except_files:
                print(pair[0], except_files)
                # load text
                contents = np.loadtxt(pair[1], skiprows=1)
                intervals = contents[:, :2]
                pitches = midi_to_hz(contents[:, 2])

                # load wav
                print('processing file {}/{}'.format(count + 1, config['files_to_load']))
                y, sr = sf.read(pair[0])
                y = librosa.core.resample(y.T, orig_sr=sr, target_sr=config['sample_rate']).T
                if config['channels'] == 1:
                    mel = wav_to_mel(config, y[:, 0])
                else:
                    print('DUAL CHANNEL NOT IMPLEMENTED')
                base_name = os.path.basename(names[i])
                data_list.append((mel.astype(np.float32), intervals.astype(np.float32), contents[:, 2].astype(np.float32), base_name))
                loaded_files.append(pair[0])
                count += 1
            if pair[0] in except_files:
                print('file skipped: ', pair[0])

                # increase count
    return data_list, loaded_files


def test_mir_evaluations(config, ground_truth):
    '''
    Sandbox function which explores the output of mir_eval
    '''
    pass


def wav_to_mel(config, y):
    '''
    Transforms the waveform into a series of mel spectrogram frames
    '''
    mel = librosa.feature.melspectrogram(
            y=y,
            sr=config['sample_rate'],
            hop_length=config['spec_hop_length'],
            fmin=config['spec_fmin'],
            n_mels=config['spec_n_bins'],
            htk=config['spec_mel_htk'],
            n_fft=config['n_fft'])
    return mel.T


def wav_to_cqt(config, y):
    '''
    Function which returns the cqt transform (using librosa)
    based on the parameters in config

    The first frame is centered at t = 0 (using zero-padding)
    '''
    cqt = librosa.core.cqt(
            y,
            sr=config['sample_rate'],
            n_bins=config['spec_n_bins'],
            hop_length=config['spec_hop_length'],
            bins_per_octave=config['cqt_bins_per_octave'],
            fmin=midi_to_hz(config['spec_fmin']))
    #freqs = librosa.core.cqt_frequencies(
    #        n_bins=config['n_bins'],
    #        fmin=fmin,
    #        bins_per_octave=config['bins_per_octave'])
    return np.abs(cqt)


def midi_to_hz(midi):
    '''
    Converts array of midi numbers to corresponding frequencies
    '''
    return 2 ** ((midi - 69) / 12) * 440


def hz_to_midi(f):
    '''
    Converts array of frequencies to corresponding midi numbers
    '''
    return 69 + 12 * np.log2(f / 440)


def frame_metrics(frame_labels, frame_predictions):
    '''
    Calculate frame-based metrics.
    '''
    frame_labels_bool = tf.cast(frame_labels, tf.bool)
    frame_predictions_bool = tf.cast(frame_predictions, tf.bool)

    frame_true_positives = tf.reduce_sum(tf.to_float(tf.logical_and(
            tf.equal(frame_labels_bool, True),
            tf.equal(frame_predictions_bool, True))))
    frame_false_positives = tf.reduce_sum(tf.to_float(tf.logical_and(
            tf.equal(frame_labels_bool, False),
            tf.equal(frame_predictions_bool, True))))
    frame_false_negatives = tf.reduce_sum(tf.to_float(tf.logical_and(
            tf.equal(frame_labels_bool, True),
            tf.equal(frame_predictions_bool, False))))
    frame_accuracy = (
            tf.reduce_sum(
                    tf.to_float(tf.equal(frame_labels_bool, frame_predictions_bool))) /
            tf.cast(tf.size(frame_labels), tf.float32))

    frame_precision = tf.where(
            tf.greater(frame_true_positives + frame_false_positives, 0),
            tf.div(frame_true_positives,
                       frame_true_positives + frame_false_positives), 
            0)
    frame_recall = tf.where(
            tf.greater(frame_true_positives + frame_false_negatives, 0),
            tf.div(frame_true_positives,
                       frame_true_positives + frame_false_negatives),
            0)
    frame_f1_score = f1_score(frame_precision, frame_recall)
    frame_accuracy_without_true_negatives = accuracy_without_true_negatives(
            frame_true_positives, frame_false_positives, frame_false_negatives)

    return {
            'true_positives': frame_true_positives,
            'false_positives': frame_false_positives,
            'false_negatives': frame_false_negatives,
            'accuracy': frame_accuracy,
            'accuracy_without_true_negatives': frame_accuracy_without_true_negatives,
            'precision': frame_precision,
            'recall': frame_recall,
            'f1_score': frame_f1_score,
            }


def f1_score(precision, recall):
    """
    Creates an op for calculating the F1 score.
    Args:
      precision: A tensor representing precision.
      recall: A tensor representing recall.
    Returns:
      A tensor with the result of the F1 calculation.
    """
    return tf.where(
            tf.greater(precision + recall, 0), 2 * ((precision * recall) / (precision + recall)), 0)


def accuracy_without_true_negatives(true_positives, false_positives, false_negatives):
    return tf.where(
            tf.greater(true_positives + false_positives + false_negatives, 0),
            true_positives / (true_positives + false_positives + false_negatives), 0)
