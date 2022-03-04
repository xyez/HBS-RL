import argparse
import os

import numpy as np
import torch as t


def main(args):
    rBs = []
    qBs = []
    pt_path = os.path.join('data', 'book', args.dataset)
    new_pt_path = os.path.join(pt_path, '-'.join(args.hmethods))
    os.makedirs(new_pt_path, exist_ok=True)
    new_file = os.path.join(new_pt_path, 'book_{}.pt'.format(np.sum(args.num_bits) * len(args.hmethods)))
    for num_bit in args.num_bits:
        for hmethod in args.hmethods:
            file = os.path.join(pt_path, hmethod, str(num_bit), 'book_{}.pt'.format(num_bit))
            data = t.load(file, map_location='cpu')

            code = data['qB'].sign()
            code[code == -1] = 0
            qBs.append(code)

            code = data['rB'].sign()
            code[code == -1] = 0
            rBs.append(code)

    rB = t.cat(rBs, dim=1)
    qB = t.cat(qBs, dim=1)
    book = {'qB': qB, 'rB': rB, 'qL': t.as_tensor(data['qL']), 'rL': t.as_tensor(data['rL'])}
    t.save(book, new_file)
    print('Over')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--hmethods', type=str, nargs='*', default=['SH', 'ITQ', 'SpH'])
    parser.add_argument('--num_bits', type=int, nargs='*', default=[16, 24, 32, 48, 64, 128])
    args = parser.parse_args()
    main(args)
