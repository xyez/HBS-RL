import argparse
import os

import numpy as np
import torch as t


def main(args):
    # num_bits=[8,16,32,64,128,256]
    rB_file = args.dataset + '_{}_rB.npy'
    rL_file = '{}_{}_rL.npy'.format(args.dataset, args.num_bits[0])
    qB_file = args.dataset + '_{}_qB.npy'
    qL_file = '{}_{}_qL.npy'.format(args.dataset, args.num_bits[0])
    npy_path = os.path.join('data', args.dataset, )
    pt_path = os.path.join('data', 'book', args.dataset)
    os.makedirs(pt_path, exist_ok=True)

    qBs = []
    rBs = []
    qL = t.as_tensor(np.load(os.path.join(npy_path, qL_file)))
    rL = t.as_tensor(np.load(os.path.join(npy_path, rL_file)))
    for i, num_bit in enumerate(args.num_bits):
        qB = t.as_tensor(np.load(os.path.join(npy_path, qB_file.format(num_bit)))).clamp_min(0)
        rB = t.as_tensor(np.load(os.path.join(npy_path, rB_file.format(num_bit)))).clamp_min(0)
        qBs.append(qB)
        rBs.append(rB)
        file_i = os.path.join(pt_path, str(num_bit), 'book_{}.pt'.format(num_bit))
        if not os.path.exists(file_i):
            os.makedirs(os.path.join(pt_path, str(num_bit)), exist_ok=True)
            t.save({'qB': qBs[-1], 'rB': rBs[-1], 'qL': qL, 'rL': rL}, file_i)
    qB = t.cat(qBs, dim=1)
    rB = t.cat(rBs, dim=1)

    t.save({'qB': qB, 'rB': rB, 'qL': qL, 'rL': rL}, os.path.join(pt_path, 'book_{}.pt'.format(sum(args.num_bits))))
    print('Over:', sum(args.num_bits))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--num_bits', type=int, nargs='*', default=[256])
    args = parser.parse_args()
    print(args)
    main(args)
