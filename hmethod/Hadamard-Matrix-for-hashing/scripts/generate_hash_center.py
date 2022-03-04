import argparse
import os
from itertools import combinations

import numpy as np
import torch as t
from scipy.linalg import hadamard  # direct import  hadamrd matrix from scipy
from scipy.special import comb  # calculate combination


def compute_hamming_distance(hash_targets, num_class=100):
    # Test average Hamming distance between hash targets
    b = list(range(num_class))
    com_num = int(comb(num_class, 2))
    c = np.zeros(com_num)
    for i in range(com_num):
        i_1 = list(combinations(b, 2))[i][0]
        i_2 = list(combinations(b, 2))[i][1]
        c[i] = sum(hash_targets[i_1] != hash_targets[i_2])
    print('distance between any two hash targets:\n', c)
    return


def single_label_hash_center_large(num_bits, num_class=100):
    ha_d = hadamard(num_bits)  # hadamard matrix
    if num_class <= num_bits:
        hash_targets = t.from_numpy(ha_d[0:num_class]).float()
        print('hash centers shape: {}'.format(hash_targets.shape))
    else:
        ha_2d = np.concatenate((ha_d, -ha_d), 0)  # can be used as targets for 2*d hash bit
        hash_targets = t.from_numpy(ha_2d[0:num_class]).float()
        print('hash centers shape: {}'.format(hash_targets.shape))
    compute_hamming_distance(hash_targets, num_class)


def generate_2_all(num_bits):
    path = 'all_{}.pt'.format(num_bits)
    if os.path.exists(path):
        data = t.load(path, map_location='cpu')
        print('load')
        return data
    num = 2 ** num_bits
    data = t.zeros((num, num_bits), dtype=t.bool, device='cpu')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data[1:num:2, 0] = 1
    for i in range(1, num_bits):
        interval = 2 ** i
        value = 0
        for j in range(0, num, interval):
            data[j:j + interval, i] = value
            value = 1 - value
    t.save(data, path)
    print('over')
    return data


def single_label_hash_center_small(num_bits, num_class=100):
    compute_dist = lambda a, b: (a ^ b).sum().item()
    target_dist = num_bits // 3
    data = generate_2_all(num_bits)
    hash_targets = t.zeros((num_class, num_bits), dtype=t.bool, device='cpu')
    num_res = 1
    hash_targets[0] = data[0]
    while (num_res < num_class):
        for data_i in data:
            well = True
            for res_i in range(num_res):
                if compute_dist(data_i, hash_targets[res_i]) < target_dist:
                    well = False
                    break
            if well:
                hash_targets[num_res] = data_i
                num_res += 1
                print(num_res)
                if num_res == num_class:
                    break
    return hash_targets


def main(args):
    num_bits = args.num_bits
    num_class = args.num_class

    if 2 * num_bits >= num_class:
        hash_targets = single_label_hash_center_large(num_bits, num_class)
    else:
        hash_targets = single_label_hash_center_small(num_bits, num_class)
    # Save the hash targets as training targets
    t.save(hash_targets, '{}_bits_{}_class.pt'.format(num_bits, num_class))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_bits', type=int, default=64)
    parser.add_argument('--num_class', type=int, default=100)
    args = parser.parse_args()
    main(args)
