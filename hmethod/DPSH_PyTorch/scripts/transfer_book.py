import argparse
import os

import torch as t


def main(args):
    # num_bits=[8,16,32,64,128,256]
    rB_file = args.dataset + '_{}_rB.npy'
    rL_file = '{}_{}_rL.npy'.format(args.dataset, args.num_bits[0])
    qB_file = args.dataset + '_{}_qB.npy'
    qL_file = '{}_{}_qL.npy'.format(args.dataset, args.num_bits[0])
    ckp_path = os.path.join('checkpoints')
    pt_path = os.path.join('data', 'book', args.dataset)
    os.makedirs(pt_path, exist_ok=True)

    ckp_files = {}
    data_files = [file for file in os.listdir(ckp_path) if args.dataset in file]
    for num_bit in args.num_bits:
        for file in data_files:
            if '_code_{}_query_'.format(num_bit) in file:
                break
        ckp_files[num_bit] = file

    qBs = []
    rBs = []
    data = t.load(os.path.join(ckp_path, list(ckp_files.values())[0]), map_location='cpu')
    qL = data['qL']
    rL = data['rL']
    for i, num_bit in enumerate(args.num_bits):
        data = t.load(os.path.join(ckp_path, ckp_files[num_bit]), map_location='cpu')
        qBs.append(data['qB'].clamp_min(0))
        rBs.append(data['rB'].clamp_min(0))
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
