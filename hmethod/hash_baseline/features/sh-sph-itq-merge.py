import os
import argparse

import numpy as np
import torch as t


def main(args):
    rBs = []
    qBs = []
    for num_bit in args.num_bits:
        for hmethod in args.hmethods:
            file = 'data/book/{}/{}/book_{}.pt'.format(args.dataset, hmethod, num_bit)
            data = t.load(file, map_location='cpu')

            code = data['qB'].sign()
            code[code == -1] = 0
            qBs.append(code)

            # code = data['vB'].sign()
            # code[code == -1] = 0
            # vBs.append(code)

            code = data['rB'].sign()
            code[code == -1] = 0
            rBs.append(code)

            # saved_dir = os.path.join('results', args.dataset, 'books', hmethod, str(num_bit))
            # os.makedirs(saved_dir, exist_ok=True)
            # saved_file = os.path.join(saved_dir, 'book_{}.pt'.format(num_bit))
            # t.save({'qB': qBs[-1], 'rB': rBs[-1], 'qL': t.as_tensor(data['qL']), 'rL': t.as_tensor(data['rL'])}, file)

    rB = t.cat(rBs, dim=1)
    qB = t.cat(qBs, dim=1)
    # vB = t.cat(vBs, dim=1)
    # labels=t.load('labels.pt',map_location='cpu')
    res = {
        'qB': qB,
        'rB': rB,
        # 'rB': t.cat((rB, vB), dim=0),
        # 'vB':vB,
        'qL': t.as_tensor(data['qL']),
        'rL': t.as_tensor(data['rL'])
    }
    # 'rL': t.cat((t.as_tensor(data['rL']), t.as_tensor(labels['vL'])), dim=0)}
    save_path = 'data/book/{}/{}'.format(args.dataset, '-'.join(args.hmethods))
    os.makedirs(save_path, exist_ok=True)
    t.save(res, os.path.join(save_path, 'book_{}.pt'.format(np.sum(args.num_bits) * len(args.hmethods))))
    print('over')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--hmethods', type=str, nargs='*', default=['SH', 'ITQ', 'SpH'])
    parser.add_argument('--num_bits', type=int, nargs='*', default=[16, 24, 32, 48, 64, 128])
    args = parser.parse_args()
    main(args)
