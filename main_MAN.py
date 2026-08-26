import datetime
import os
import argparse
import time

import torch
from iCaRL_MAN import iCaRLmodel
from models_MAN import vgg_c


def parse_args():
    parser = argparse.ArgumentParser(description='iCaRL Incremental Learning')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Root directory path of the dataset (required)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory to save model checkpoints (default: checkpoint)')
    parser.add_argument('--numclass', type=int, default=1,
                        help='Number of original task classes  (default: 1)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--task_size', type=int, default=1,
                        help='Number of classes contained in each task (default: 1)')
    parser.add_argument('--memory_size', type=int, default=150,
                        help='Number of samples in memory (default: 150)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs per task (default: 300)')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate (default: 1e-5)')
    parser.add_argument('--task', type=int, default='',
                        help='Total number of incremental tasks ')
    return parser.parse_args()


def main():
    args = parse_args()
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    start_time_total = time.time()
    feature_extractor = vgg_c.vgg19_trans()

    model = iCaRLmodel(
        numclass=args.numclass,
        feature_extractor=feature_extractor,
        batch_size=args.batch_size,
        task_size=args.task_size,
        memory_size=args.memory_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        dataset_dir=args.dataset_dir
    )

    for i in range(args.task):
        model.beforeTrain()
        model.train()
        model.afterTrain()

    end_time_total = time.time()
    print('Total training time: {:.2f} hours'.format((end_time_total - start_time_total) / 3600))
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


if __name__ == '__main__':
    main()