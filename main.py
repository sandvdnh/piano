import argparse
import os
import yaml
import tensorflow as tf
from lib.utils import read_files
from lib.generate_labels import onset


def main(config, args):
    data_list = read_files(config)
    onset_label = onset(config, data_list[0])
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='dev')
    args = parser.parse_args()

    # load config file
    config = yaml.safe_load(open('./config/' + args.config + '.yml'))
    main(config, args)

