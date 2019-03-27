import argparse
import os
import yaml
import tensorflow as tf
from lib.dataset import create_dataset
from src.model import Model


def main(config, args):
    dataset = create_dataset(config)
    iterator = dataset.make_one_shot_iterator()
    mel, onset_labels, frame_labels, weights = iterator.get_next()
    is_training = tf.constant(True, dtype=tf.bool)
    model = Model(config, mel, is_training)
    node = model.onset_output

    with tf.Session() as sess:
        sess.run(tf.initializers.global_variables())
        for i in range(1):
            a = sess.run(node)
            print(a.shape)
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
    parser.add_argument('--config', default='dev')
    args = parser.parse_args()

    # load config file
    config = yaml.safe_load(open('./config/' + args.config + '.yml'))
    main(config, args)

