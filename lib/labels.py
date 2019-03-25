from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os


def onset(config, raw_data):
    '''
    generates note onset ground truth for the tuple raw_data: (mel, intervals, pitches)
    returns an 88 by n_frames binary array
    '''
    mel = raw_data[0]
    intervals = raw_data[1]
    pitches = raw_data[2]

    n_frames = mel.shape[0]
    onset_labels = np.zeros((88, n_frames))

    # convert time axis to samples
    intervals = intervals * config['sample_rate']
    # 'time' axis for each mel spec frame
    frame_samples = config['spec_hop_length'] * np.arange(n_frames)
    # length of each note
    note_lengths = (intervals[:, 1] - intervals[:, 0])
    fixed = config['onset_length'] / 1000 * config['sample_rate'] * np.ones(len(note_lengths))
    # assume a max onset of config['onset_length']
    onset_lengths = np.minimum(note_lengths, fixed) # note lengths (in samples)
    # onset_lengths /= config['spec_hop_length'] + 1 # onset lengths (in frames)
    for i, interval in enumerate(intervals):
        start = int(np.argmax(frame_samples + config['n_fft'] / 2 > interval[0]))
        #stop = start + int(np.ceil(onset_lengths[i]))
        stop = int(np.argmax(frame_samples - config['n_fft'] / 2 > interval[0] + onset_lengths[i]))
        onset_labels[int(pitches[i] - config['spec_fmin']), start:stop] = 1
    return onset_labels


def create_data_entry(config, raw_data):
    '''
    returns dictionary with preprocessed training data used to write tfrecords files
    '''
    onset_labels = onset(config, raw_data)
    frame_labels, weights = frame(config, raw_data)
    entry = {
            'name': raw_data[3],
            'mel': raw_data[0].astype(np.float32),
            'onset_labels': onset_labels.astype(np.float32),
            'frame_labels': frame_labels.astype(np.float32),
            'weights': weights.astype(np.float32),
            }
    return entry


def frame(config, raw_data):
    '''
    generates frame ground truth for the tuple raw_data: (mel, intervals, pitches)
    '''
    mel = raw_data[0]
    intervals = raw_data[1]
    pitches = raw_data[2]

    n_frames = mel.shape[0]
    frame_labels = np.zeros((88, n_frames))

    # convert time axis to samples
    intervals = intervals * config['sample_rate']
    # 'time' axis for each mel spec frame
    frame_samples = config['spec_hop_length'] * np.arange(1, n_frames + 1)
    # length of each note
    note_lengths = (intervals[:, 1] - intervals[:, 0])
    for i, interval in enumerate(intervals):
        start = int(np.argmax(frame_samples + config['n_fft'] / 2 > interval[0]))
        stop = int(np.argmax(frame_samples - config['n_fft'] / 2 > interval[1]))
        frame_labels[int(pitches[i] - config['spec_fmin']), start:stop] = 1
    return frame_labels, 0
