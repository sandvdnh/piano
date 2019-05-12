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
    onset_labels = np.zeros((n_frames, 88))

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
        #onset_labels[int(pitches[i] - config['spec_fmin']), start:stop] = 1
        onset_labels[start:stop, int(pitches[i] - config['spec_fmin'])] = 1
    return onset_labels


def create_data_entry(config, raw_data):
    '''
    returns dictionary with preprocessed training data used to write tfrecords files
    '''
    print('CREATING DATA ENTRY')
    onset_labels = onset(config, raw_data)
    frame_labels, weights = frame(config, raw_data)

    mel = raw_data[0]
    mel_ = np.zeros((mel.shape[0], 5, mel.shape[1]))
    fix = np.zeros((2, mel.shape[1]))
    mel = np.concatenate(
            (fix, mel, fix),
            axis = 0)

    for i in range(mel_.shape[0]):
        mel_[i, :, :] = mel[i : i + 5, :].copy()

    #fix = np.zeros((2, onset_labels.shape[1]))
    #onset_labels = np.concatenate(
    #        (fix, onset_labels, fix),
    #        axis = 0)

    #fix = np.zeros((2, frame_labels.shape[1]))
    #frame_labels = np.concatenate(
    #        (fix, frame_labels, fix),
    #        axis = 0)

    #fix = np.zeros((2, weights.shape[1]))
    #weights = np.concatenate(
    #        (fix, weights, fix),
    #        axis = 0)

    entry = {
            'name': raw_data[3],
            'mel': mel_.astype(np.float32),
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
    frame_labels = np.zeros((n_frames, 88))

    # convert time axis to samples
    intervals = intervals * config['sample_rate']
    # 'time' axis for each mel spec frame
    frame_samples = config['spec_hop_length'] * np.arange(1, n_frames + 1)
    # length of each note
    note_lengths = (intervals[:, 1] - intervals[:, 0])
    starts = []
    stops = []
    for i, interval in enumerate(intervals):
        start = int(np.argmax(frame_samples + config['n_fft'] / 2 > interval[0]))
        starts.append(start)
        stop = int(np.argmax(frame_samples - config['n_fft'] / 2 > interval[1]))
        stops.append(stop)
        #frame_labels[int(pitches[i] - config['spec_fmin']), start:stop] = 1
        frame_labels[start:stop, int(pitches[i] - config['spec_fmin'])] = 1
    return frame_labels, np.zeros(frame_labels.shape)

