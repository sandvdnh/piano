from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.data.python.ops import sliding
import os
import glob
from .utils import read_files
from .labels import create_data_entry


def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def wrap_int64_list(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def to_tfrecords(config, entry):
    '''
    writes an entry to a tfrecords file stored in config['cache']/filename
    '''
    filename = entry['name'] + '.tfrecords'
    path = os.path.join(config['cache'], filename)
    with tf.python_io.TFRecordWriter(path) as writer:
        # create dict
        data_dict = {
                'name': wrap_bytes(entry['name'].encode('utf-8')),
                'mel': wrap_bytes(entry['mel'].tostring()),
                'mel_shape': wrap_int64_list(entry['mel'].shape),
                'onset_labels': wrap_bytes(entry['onset_labels'].tostring()),
                'onset_shape': wrap_int64_list(entry['onset_labels'].shape),
                'frame_labels': wrap_bytes(entry['frame_labels'].tostring()),
                'frame_shape': wrap_int64_list(entry['frame_labels'].shape),
                'weights': wrap_bytes(entry['weights'].tostring()),
                'weights_shape': wrap_int64_list(entry['weights'].shape),
                }
        example = tf.train.Example(features=tf.train.Features(feature=data_dict))
        writer.write(example.SerializeToString())
    return path


def _parser(record):
    '''
    parser function for the TFRecordDataset object
    '''
    features = {
            'name': tf.FixedLenFeature([1], tf.string),
            'mel': tf.FixedLenFeature([1], tf.string),
            'mel_shape': tf.FixedLenFeature([3], tf.int64),
            'onset_labels': tf.FixedLenFeature([1], tf.string),
            'onset_shape': tf.FixedLenFeature([2], tf.int64),
            'frame_labels': tf.FixedLenFeature([1], tf.string),
            'frame_shape': tf.FixedLenFeature([2], tf.int64),
            'weights': tf.FixedLenFeature([1], tf.string),
            'weights_shape': tf.FixedLenFeature([2], tf.int64),
            }
    parsed = tf.parse_single_example(record, features)
    name = parsed['name']
    mel = tf.decode_raw(parsed['mel'], tf.float32)
    mel_shape = parsed['mel_shape']
    mel = tf.reshape(mel, shape=mel_shape)
    onset_labels = tf.decode_raw(parsed['onset_labels'], tf.float32)
    onset_shape = parsed['onset_shape']
    onset_labels = tf.reshape(onset_labels, shape=onset_shape)
    frame_labels = tf.decode_raw(parsed['frame_labels'], tf.float32)
    frame_shape = parsed['frame_shape']
    frame_labels = tf.reshape(frame_labels, shape=frame_shape)
    weights = tf.decode_raw(parsed['weights'], tf.float32)
    weights_shape = parsed['weights_shape']
    weights = tf.reshape(weights, shape=weights_shape)
    return mel, onset_labels, frame_labels, weights


def _apply_window(mel, onset_labels, frame_labels, weights):
    '''
    create dataset from tuple of tensors
    NOT USED
    '''
    dataset = tf.data.Dataset.from_tensor_slices((mel, onset_labels, frame_labels, weights))
    #dataset = dataset.window(5, 1, 1, True)
    dataset = dataset.apply(sliding.sliding_window_batch(5, 1))
    dataset = dataset.batch(1000000)
    return dataset


def create_dataset(config):
    '''
    returns a TFRecordDataset object from #(files_to_load) files from config['path']
    '''
    filenames = []
    if config['create_tfrecords']:
        data_list = read_files(config)
        for raw_entry in data_list:
            entry = create_data_entry(config, raw_entry)
            filenames.append(to_tfrecords(config, entry))
    else:
        path = os.path.join(config['cache'], '*.tfrecords')
        filenames = glob.glob(path)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parser)
    #dataset = dataset.flat_map(_apply_window)
    #dataset = dataset.batch(100)
    dataset = dataset.repeat()
    return dataset
