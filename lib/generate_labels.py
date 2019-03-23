from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import mir_eval
import glob
import os


def onset(config, data_entry):
    '''
    generates note onset ground truth for the tuple data_entry: (mel, intervals, pitches)
    returns an 88 by n_frames binary array
    '''
    mel = data_entry[0]
    intervals = data_entry[1]
    pitches = data_entry[2]


    n_frames = mel.shape[0]
    onset_labels = np.zeros((88, n_frames))

    # sort intervals by onset time
    indices = np.argsort(intervals[:, 0])
    intervals = intervals[indices]
    pitches = pitches[indices]

    # convert time axis to samples
    intervals_ = intervals * config['sample_rate']
    # 'time' axis for each mel spec frame
    frame_samples = config['spec_hop_length'] * np.arange(1, n_frames + 1)
    # length of each note onset
    note_lengths = (intervals[:, 1] - intervals[:, 0]) * config['sample_rate']
    fixed = 0.032 * config['sample_rate'] * np.ones(len(note_lengths))
    note_lengths = np.minimum(note_lengths, fixed)

    return onset_labels

