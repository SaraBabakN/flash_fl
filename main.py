
import os
import torch
import random
import argparse
import numpy as np
import torch.cuda as cuda

from training.manager import System
from constants import DENSE, SPDST, JMWST, PDST, TRUE, TINY
from constants import CIFAR10, MNIST, CIFAR100, RESNET18, model_choices, experiment_choices


def start(parser):
    args = parser.parse_args()
    seed, gpu = args.seed, args.gpu
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda.manual_seed_all(seed)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    args.conv_only = args.conv_only == TRUE
    if args.experiment_type in [DENSE, PDST]:
        args.update_mode = 0
    if args.hetero_client:
        assert args.experiment_type in [JMWST, SPDST]
        assert args.subsample_method != 'avg'
    if args.model_type == RESNET18:
        assert args.conv_only
    if args.dataset in [CIFAR10, MNIST]:
        args.num_classes = 10
    elif args.dataset == CIFAR100:
        args.num_classes = 100
    elif args.dataset == TINY:
        args.num_classes = 200
    args.lr_gamma = np.log(0.001 / args.lr_start)
    linear = '_linear' if not args.conv_only else ''
    data_name = args.dataset
    args.mask_key_path = f'keys/{args.model_type}{linear}_{data_name}_mask.pt'
    if args.remove_first_last:
        args.mask_key_path = f'keys/{args.model_type}{linear}_{data_name}_mask_middle.pt'
    return args


parser = argparse.ArgumentParser()
# general
parser.add_argument('--gpu', default='6')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--lr_start', type=float, default=0.1)
parser.add_argument('--local_bs', type=int, default=32)
parser.add_argument('--model_type', type=str, default=RESNET18, choices=model_choices)
parser.add_argument('--dataset', type=str, default=CIFAR10, choices=[CIFAR10, CIFAR100, MNIST, TINY])
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--local_epoch', type=int, default=1)
parser.add_argument('--frac', type=float, default=0.1)
parser.add_argument('--init_frac', type=float, default=0.1)
parser.add_argument('--init_epoch', type=int, default=10)
parser.add_argument('--fl_rounds', type=int, default=600)
parser.add_argument('--update_mode', type=int, default=0, choices=[0, 1], help="0: fedAvg, 1:FMA")
parser.add_argument('--num_users', type=int, default=100)
parser.add_argument('--experiment_type', default=JMWST, type=str, choices=experiment_choices)
parser.add_argument('--hetero_client', action='store_true', default=False)
parser.add_argument('--density', type=float, default=0.05)
parser.add_argument('--prune_rate', type=float, default=0.25)
parser.add_argument('--conv_only', type=str, default=TRUE)
parser.add_argument('--remove_first_last', action='store_true', default=False)
parser.add_argument('--subsample_method', type=str, default='avg', choices=['aggregate', 'avg'])
parser.add_argument('--path', type=str, default='data')
parser.add_argument('--jmwst_update_interval', type=int, default=1)


if __name__ == "__main__":
    args = start(parser)
    server = System(args)
    server.start_federated_learning()
