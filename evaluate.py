import argparse
import os
import random

import numpy as np
import torch as t

from utils.get_results import get_results, show_results


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    books = {}
    for method in args.method:
        if type(method) == str:
            book = t.load(os.path.join(args.book_prefix, args.dataset, method, str(args.num_bit), 'book_{}.pt'.format(args.num_bit)), map_location='cpu')
            books[method] = {k: v.float() for k, v in book.items() if k in ['rB', 'rL', 'qB', 'qL']}
        elif type(method) == list:
            hash_method = method[0]
            hash_method_bit = int(method[1])
            book = t.load(os.path.join(args.book_prefix, args.dataset, hash_method, 'book_{}.pt'.format(hash_method_bit)), map_location='cpu')
            book = {k: v.float() for k, v in book.items() if k in ['rB', 'rL', 'qB', 'qL']}
            for method_i in method[2:]:
                index = t.load(os.path.join(args.index_prefix, args.dataset, method[0], method_i, 'ind_{}_{}.pt'.format(hash_method_bit, args.num_bit)), map_location='cpu')
                book_i = {'qB': book['qB'][:, index], 'rB': book['rB'][:, index], 'qL': book['qL'], 'rL': book['rL']}
                books['{}_{}_{}_{}'.format(hash_method, hash_method_bit, method_i, args.num_bit)] = book_i

    results = get_results(books, args.device, args.topK, num_interval=args.num_interval, float_type=args.float)
    for key, value in results.items():
        print('{:>10s}: {:>30s}: {:>8.2f}'.format(args.dataset, key, value['map'] * 100))
    if args.plot:
        show_results(args.dataset, results)
    # print('Over')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--method', type=str, default='lsh,lsh{Random_NDomSet_BSPPO}512')
    parser.add_argument('--num_bit', type=int, default=32)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--topK', type=int, default=50000)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--plot', type=eval, default=True)
    parser.add_argument('--float', type=int, default=2, choices=[1, 2])
    parser.add_argument('--num_interval', type=int, default=50)

    # default
    parser.add_argument('--book_prefix', type=str, default='data/book')
    parser.add_argument('--index_prefix', type=str, default='data/index')
    args = parser.parse_args()
    method_old = args.method.split(',')
    method_new = []
    for method_i in method_old:
        index0 = method_i.find('{')
        if index0 < 0:
            method_new.append(method_i)
        else:
            method_i_new = []
            index1 = method_i.find('}')
            method_i_new.append(method_i[:index0])
            method_i_new.append(method_i[index1 + 1:])
            method_i_new += method_i[index0 + 1:index1].split('_')
            method_new.append(method_i_new)
    # print(method_new)
    # print('hhehh')
    # raise NotImplementedError()
    args.method = method_new
    main(args)
