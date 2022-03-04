import random

import torch as t


def test(args, ind_file):
    index = t.as_tensor(random.sample(range(args.num_bit), args.num_subbit), dtype=t.int64)
    index = t.zeros(args.num_bit, dtype=t.bool).scatter(0, index, 1)
    # print('Random\n', t.where(index)[0])
    t.save(index, ind_file)
    # print('Random Over')


if __name__ == '__main__':
    num_sub_bits = [16, 24, 32, 48, 64, 128]
    num_bits = sum(num_sub_bits)
    start_idx = 0
    for num_sub_bit in num_sub_bits:
        end_idx = start_idx + num_sub_bit
        index = t.arange(start_idx, end_idx)
        start_idx = end_idx
        index = t.zeros(num_bits, dtype=t.bool).scatter(0, index, 1)
        t.save(index, 'ind_{}_{}.pt'.format(num_bits, num_sub_bit))
    # print('Over')
