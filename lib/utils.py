from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator
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

def plot_labels(result, config):
    '''
    function that generates an output image of part of the output
    to evaluate the performance of the model
    '''
    # first try on 1 sequence, then extend
    frame_output_list = result['frame_output']
    frame_labels_list = result['frame_labels']
    batch_size = config['batch_size']
    test_iters = config['test_iters']
    sequence_length = config['sequence_length']
    frame_labels = np.ones((0, sequence_length, 88)) 
    frame_output = np.zeros((0, sequence_length, 88)) 
    for frame_label_ in frame_labels_list:
        frame_labels = np.concatenate((frame_labels, frame_label_.copy()), axis=0)
    frame_labels = np.reshape(frame_labels, (batch_size * test_iters * sequence_length, 88))
    for frame_output_ in frame_output_list:
        frame_output = np.concatenate((frame_output, frame_output_.copy()), axis=0)
    frame_output = np.reshape(frame_output, (batch_size * test_iters * sequence_length, 88))

    frame_output *= 2 
    #rows = frame_output.shape[0]
    #n = frame_output.shape[0]
    n = frame_labels.shape[0]
    x_axis = np.arange(n) * config['spec_hop_length'] / config['sample_rate']
    #x_axis = np.arange(n)

    fig = plt.figure(figsize=(3.5, 2.4))
    ax0 = fig.add_subplot(111)
    color = ['k', 'g', 'r', 'k', 'cyan', 'm', 'y']
    #frame_labels = frame_labels[:150, :]
    for k in range(88):
        values = k * (frame_output[:, k] > 1/2)
        pos = np.where(np.abs(np.diff(values)) > 1)[0]
        values = values.astype(np.float32)
        if len(pos) > 0:
            values[pos] = float('NaN')
        ax0.plot(x_axis, values + config['spec_fmin'], linewidth=2, color=color[1], alpha=0.8)
        #print(values)
        #x_list, y_list = _helper(x_axis, values)
        #for i, x in enumerate(x_list):
        #    ax0.plot(x, y_list[i] + config['spec_fmin'], linewidth=2, color=color[1], alpha=0.5)
        values = k * (frame_labels[:, k] > 1/2)
        pos = np.where(np.abs(np.diff(values)) > 1)[0]
        values = values.astype(np.float32)
        if len(pos) > 0:
            values[pos] = float('NaN')
        ax0.plot(x_axis, values + config['spec_fmin'], linewidth=2, color=color[0], alpha=0.4)
        #print(values)
        #x_list, y_list = _helper(x_axis, values)
        #for i, x in enumerate(x_list):
        #    ax0.plot(x, y_list[i] + config['spec_fmin'], linewidth=2, color=color[0], alpha=0.5)

    ax0.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax0.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax0.get_yaxis().set_tick_params(which='both', direction='in')
    ax0.get_xaxis().set_tick_params(which='both', direction='in')
    ax0.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax0.set_xlim([np.min(x_axis), np.max(x_axis)])
    ax0.set_ylim([0, 88])
    ax0.set_xlabel('Time [s]')
    ax0.set_ylabel('MIDI note')
    fig.savefig('./tmp/' + config['name'] + '.pgf', bbox_inches='tight')
    fig.savefig('./tmp/' + config['name'] + '.pdf', bbox_inches='tight')
    return 0

def _helper(x_axis, values):
    indices = np.where(values)[0]
    x_list = []
    y_list = []
    if len(indices) > 0:
        original = indices.copy()
        while len(original) > 0:
            indices = original
            while indices[-1] - indices[0] + 1 != len(indices):
                indices = indices[:-1]
            original = original[len(indices):]
            x_ = x_axis[indices]
            y_axis = values[indices]
            x_list.append(x_)
            y_list.append(y_axis)
    return x_list, y_list

def __helper(x_axis, values):
    '''
    identifies groups, returns them as list
    '''
    i = 0
    x_list = []
    y_list = []
    indices = []
    new = True
    start = 0
    previous_length = 1
    while i < 50:
        #print(i)
        #start = i - len(indices)
        i += 1
        count = np.count_nonzero(values[start:i])
        if count > 0 and new:
            #print(indices)
            print(count)
            print('new group at ', i)
            print(values[i-1:i+4])
            start = i - 1
            previous_length = 1
            new = False
        else:
            if count == previous_length + 1:
                print(count)
                previous_length += 1
            elif count == previous_length:
                stop = i - 1
                x_list.append(x_axis[start:stop])
                y_list.append(values[start:stop]) 
                new = True
                start = i
        #print(start, i, previous_length, values[start:i + 1])
    #print('DONE')
    return x_list, y_list




def accuracy_without_true_negatives(true_positives, false_positives, false_negatives):
    return tf.where(
            tf.greater(true_positives + false_positives + false_negatives, 0),
            true_positives / (true_positives + false_positives + false_negatives), 0)
