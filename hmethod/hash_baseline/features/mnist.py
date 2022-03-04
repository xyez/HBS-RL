import argparse
import os

import scipy.io
import torch as t


def get_feature(mat_file, device):
    mat = scipy.io.loadmat(mat_file)
    train_feature = mat['MNIST_trndata']  # (69000,784) [-1,1]
    test_feature = mat['MNIST_tstdata']  # (1000,784) [-1,1]
    train_label_idx = mat['MNIST_trnlabel'][0]  # (69000,)
    test_label_idx = mat['MNIST_tstlabel'][0]  # (69000,)

    train_feature = t.as_tensor(train_feature, dtype=t.float32, device=device)
    test_feature = t.as_tensor(test_feature, dtype=t.float32, device=device)
    train_label_idx = t.as_tensor(train_label_idx, dtype=t.int64, device=device)
    test_label_idx = t.as_tensor(test_label_idx, dtype=t.int64, device=device)
    train_label = t.zeros((train_feature.size(0), 10), dtype=t.float32, device=device)
    test_label = t.zeros((test_feature.size(0), 10), dtype=t.float32, device=device)
    train_label.scatter_(1, train_label_idx.unsqueeze(-1), 1)
    test_label.scatter_(1, test_label_idx.unsqueeze(-1), 1)
    feature = {'rF': train_feature, 'qF': test_feature, 'rL': train_label, 'qL': test_label}
    return feature


def get_book(feature, num_bit):
    # normalize
    train_feature_mean = feature['rF'].mean(dim=0, keepdim=True)
    train_feature_std = feature['rF'].std(dim=0, keepdim=True)
    train_feature = (feature['rF'] - train_feature_mean) / (train_feature_std + 1e-6)
    test_feature = (feature['qF'] - train_feature_mean) / (train_feature_std + 1e-6)

    gaussian_weight = t.randn((train_feature.size(1), num_bit), dtype=t.float32, device=feature['rF'].device)
    train_data = train_feature @ gaussian_weight
    train_data = (train_data > 0).float()
    test_data = test_feature @ gaussian_weight
    test_data = (test_data > 0).float()

    book = {'qB': test_data, 'rB': train_data, 'qL': feature['qL'], 'rL': feature['rL']}
    return book


def main(args):
    feature_path = 'data/feature/mnist/'
    os.makedirs(feature_path, exist_ok=True)
    feature_file = os.path.join(feature_path, 'feature.pt')

    book_path = 'data/book/mnist/LSH/'
    os.makedirs(book_path, exist_ok=True)
    book_file = os.path.join(book_path, 'book_{}.pt')

    if os.path.exists(feature_file):
        feature = t.load(feature_file, map_location=args.device)
    else:
        mat_file = os.path.join(args.file_path, 'data_mnist', args.mat_file)
        feature = get_feature(mat_file, args.device)
        t.save(feature, feature_file)

    for num_bit in args.num_bits:
        book = get_book(feature, num_bit)
        t.save(book, book_file.format(num_bit))

    print('Over')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_bits', type=int, nargs='*', default=[512])

    parser.add_argument('--mat_file', type=str, default='MNIST_gnd_release.mat')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    args.file_path = os.path.dirname(__file__)
    main(args)
