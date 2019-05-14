import argparse
import os
import yaml
import numpy as np
from lib.dataset import create_dataset
from lib.utils import plot_labels
from src.model import Model
from src.model import Trainer
import tensorflow as tf


def main(config, args):
    dataset, loaded_files = create_dataset(config, is_training=True)
    iterator = tf.data.Iterator.from_structure(dataset.output_types)
    init_train_iterator = iterator.make_initializer(dataset)
    #iterator = dataset.make_one_shot_iterator()
    mel, onset_labels, frame_labels, weights = iterator.get_next()
    #with tf.Session() as sess:
    #    feed = {}
    #    sess.run(tf.initializers.global_variables())
    #    sess.run(init_train_iterator)
    #    for i in range(1):
    #        a, b, c, d = sess.run([mel, onset_labels, frame_labels, weights], feed_dict=feed)
            #print('shape of mel after iterator: ', a.shape)
            #print('first entry: ', a[0, 0, :, :, 0])
            #print('shape of labels: ', c.shape)
            #print('first entry: ', c[0, 0, :])
    trainer = Trainer(
            config=config,
            input_=mel,
            onset_labels=onset_labels,
            frame_labels=frame_labels,
            weights=None)
    #trainer.train(init_train_iterator)

    config['create_tfrecords'] = False
    test_dataset, _ = create_dataset(config, is_training=False, except_files=loaded_files)
    init_test_iterator = iterator.make_initializer(test_dataset)
    result = trainer.test(init_test_iterator, cache_output=True)
    plot_labels(result, config)


#    is_training = tf.placeholder(tf.bool)
#    reset_state = tf.placeholder(tf.bool)
#    model = Model(config, mel, is_training, reset_state)
#    node = model.onset_output
#    node_ = model.frame_output


    #with tf.Session() as sess:
    #    feed = {is_training: True, reset_state: False}
    #    sess.run(tf.initializers.global_variables())
    #    for i in range(1):
    #        a, b, c, d = sess.run([mel, onset_labels, frame_labels, weights], feed_dict=feed)
    #        print(a.shape)
    #        print(b.shape)
    #        print(c.shape)
    #        print(d.shape)
    #        e, f = sess.run([node, node_], feed_dict=feed)
    #        print(e.shape)
    #        print(f.shape)
    #input_, ground_truth = iterator.get_next()
    #output = model(input_)
    #loss = create_loss(output, ground_truth)
    #train_op = optimizer.minimize(loss)

    #with tf.Session() as sess:
    #    writer = tf.summary.FileWriter('./logs/logs', sess.graph)
    #    writer.close()
    #with tf.Session() as sess:
    #    a = sess.run(node)
    #    print(a.shape)
    #    for i in range(3):
    #        _, r_loss = sess.run([train_op, loss])
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='dev1')
    args = parser.parse_args()

    # load config file
    config = yaml.safe_load(open('./config/' + args.config + '.yml'))
    main(config, args)

