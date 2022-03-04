import argparse
import os

import torch as t


def merge_book(args):
    rBs = []
    qBs = []

    rL = 0
    qL = 0
    for num_bit in args.num_bits:
        file = os.path.join(str(num_bit), 'book_{}.pt'.format(num_bit))
        data = t.load(file, map_location='cpu')
        rBs.append(data['rB'])
        qBs.append(data['qB'])
        rL = data['rL']
        qL = data['qL']
    for i in range(-1, len(args.num_bits)):
        if i == -1:
            rB = rBs
            qB = qBs
            num_bit = sum(args.num_bits)
        else:
            rB = rBs[:i] + rBs[i + 1:]
            qB = qBs[:i] + qBs[i + 1:]
            num_bit = sum(args.num_bits[:i] + args.num_bits[i + 1:])
        rB = t.cat(rB, dim=1)
        qB = t.cat(qB, dim=1)
        book = {'rB': rB, 'qB': qB, 'rL': rL, 'qL': qL}
        t.save(book, 'book_{}.pt'.format(num_bit))
    print('Over')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_bits', type=int, nargs='*', default=[16, 24, 32, 48, 64, 128])
    args = parser.parse_args()

    merge_book(args)
