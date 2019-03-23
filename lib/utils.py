from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import mir_eval
import glob
import os
import librosa
import soundfile as sf

from scipy.io.wavfile import read


def read_files(config):
    '''
    Reads in text files in config['path'] and returns an (n, 2) ndarray
    with intervals and an (n,) ndarray with pitches in Hz

    returns a list of tuples (a, b, c), where each tuple corresponds to a file:
    - a: cqt transform of .wav file
    - b: list of intervals from .txt file
    - c: corresponding list of pitches (in MIDI)
    '''
    data_list = []
    dirs = config['dirs'] # list of names of subdirectories in MAPS dataset to be included
    file_list = []
    for _ in dirs:
        path = os.path.join(config['path'], _, 'MUS')
        path = os.path.join(path, '*.wav')
        wav_files = glob.glob(path)
        for wav_file in wav_files:
            name, _ = os.path.splitext(wav_file)
            txt_file = name + '.txt'
            file_list.append((wav_file, txt_file))
        count = 0
        for pair in file_list:
            if count < config['files_to_load']:
                # load text
                contents = np.loadtxt(pair[1], skiprows=1)
                intervals = contents[:, :2]
                pitches = midi_to_hz(contents[:, 2])

                # load wav
                y, sr = sf.read(pair[0])
                if sr != config['sample_rate']:
                    print('Resampling...')
                if config['channels'] == 1:
                    mel = wav_to_mel(config, y[:, 0])
                else:
                    print('DUAL CHANNEL NOT IMPLEMENTED')
                data_list.append((mel, intervals, contents[:, 2]))

                # increase count
                count += 1
    return data_list


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
            y,
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
