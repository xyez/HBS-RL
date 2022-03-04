import argparse
import importlib
import os
import random
import sys

import numpy as np
import torch as t


def main(args):
    args.cwd = os.getcwd()
    random.seed(args.seed)
    np.random.seed(args.seed)
    t.random.manual_seed(args.seed)
    if args.device != 'cpu':
        t.cuda.random.manual_seed(args.seed)

    if args.test:
        # print('Test...')
        test = importlib.import_module('bitselect.{}.test'.format(args.method)).test
        os.makedirs(args.index_prefix, exist_ok=True)
        ind_file = os.path.join(args.index_prefix, 'ind_{}_{}.pt'.format(args.num_bit, args.num_subbit))
        test(args, ind_file)
    else:
        # print('Train...')
        train = importlib.import_module('bitselect.{}.train'.format(args.method)).train
        os.makedirs(args.exp_path, exist_ok=True)
        command_file = os.path.join(args.exp_path, 'run_command.sh')
        with open(command_file, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write(' '.join(['python3', sys.argv[0]] + [i if i.startswith('--') else "'{}'".format(i) for i in sys.argv[1:]]))
        os.chmod(command_file, 0o755)
        train(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Env Parameters
    parser.add_argument('--subref_size', type=int, default=5000)
    parser.add_argument('--subval_size', type=int, default=2000)
    parser.add_argument('--subtest_size', type=int, default=10)
    # select method
    parser.add_argument('--method', type=str, default='PPO', choices=['PPO', 'NDomSet', 'Random'])
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--hmethod', type=str, default='SH-SpH-ITQ')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_bit', type=int, default=1512)
    parser.add_argument('--num_subbit', type=int, default=32)
    parser.add_argument('--model_prefix', type=str, default='data/checkpoint')
    parser.add_argument('--index_prefix', type=str, default='data/index')
    parser.add_argument('--seed', type=int, default=123)

    # NDomSet Parameters
    parser.add_argument('--num_worker', type=int, default=32)
    parser.add_argument('--num_train', type=int, default=3000)
    parser.add_argument('--num_same_neighbor', type=int, default=30)
    parser.add_argument('--gamma', type=float, default=0.25)
    parser.add_argument('--lambda0', type=float, default=1)
    parser.add_argument('--feature_prefix', type=str, default='data/feature')
    parser.add_argument('--book_prefix', type=str, default='data/book')
    parser.add_argument('--book_suffix', type=str, default='')
    parser.add_argument('--use_matlab', type=eval, default=False)
    # PPO Parameters
    parser.add_argument('--exp_name', type=str, default='base')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--discount_factor', default=0.99)
    parser.add_argument('--rho', type=float, default=0.2)
    parser.add_argument('--num_train_pi', type=int, default=80)
    parser.add_argument('--num_train_v', type=int, default=80)
    parser.add_argument('--target_kl', type=float, default=0.01)
    parser.add_argument('--ent_coef', type=float, default=0.01)
    parser.add_argument('--clip_v', type=eval, default=False)
    parser.add_argument('--use_test', type=eval, default=False)
    parser.add_argument('--reward_step', type=int, default=0)
    parser.add_argument('--num_env', type=int, default=64)
    parser.add_argument('--num_epoch_step', type=int, default=128)
    parser.add_argument('--num_episode', type=int, default=int(1e5))
    parser.add_argument('--trainset_size', type=int, default=0)
    parser.add_argument('--testset_size', type=int, default=1000)

    parser.add_argument('--use_candi', type=eval, default=False)
    parser.add_argument('--num_val_times', type=int, default=3)
    parser.add_argument('--num_candi', type=int, default=2)

    parser.add_argument('--test', type=eval, default=True)
    # test param
    args = parser.parse_args()
    args.cwd = os.getcwd()
    args.model_prefix = os.path.join(args.cwd, args.model_prefix, args.dataset, args.hmethod, args.method + '-{}'.format(args.exp_name))
    args.index_prefix = os.path.join(args.cwd, args.index_prefix, args.dataset, args.hmethod, args.method)
    args.feature_prefix = os.path.join(args.cwd, args.feature_prefix, args.dataset)
    args.book_prefix = os.path.join(args.cwd, args.book_prefix, args.dataset, args.hmethod, args.book_suffix)
    args.exp_name = '{}-{}-{}'.format(args.exp_name, args.num_bit, args.num_subbit)
    bit_name = '{}-{}'.format(args.num_bit, args.num_subbit)
    args.exp_path = os.path.join(args.model_prefix, bit_name, bit_name + '_s{}'.format(args.seed))
    main(args)
