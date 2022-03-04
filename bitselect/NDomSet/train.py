import os
import random

import numpy as np
import torch as t
import torch.multiprocessing as mp


def compute_edge(i, trainbook, num_bits):
    res = []
    temp_ind = t.ones(5, dtype=t.bool)
    temp_ind[2] = False
    bits_i = trainbook[:, i]
    num_train=trainbook.size(0)
    for j in range(i + 1, num_bits):
        bits_j = trainbook[:, j]
        x_hist = bits_i.histc(bins=2, min=0, max=2) / num_train
        y_hist = bits_j.histc(bins=2, min=0, max=2) / num_train
        bits_ij = bits_i + 3 * bits_j
        res_hist = bits_ij.histc(bins=5, min=0, max=5) / num_train
        xy_hist = t.stack((t.stack((res_hist[0], res_hist[1] + res_hist[2])), res_hist[3:])).t()
        single_xy = x_hist.unsqueeze(1) * y_hist.unsqueeze(0)
        value = (xy_hist * (xy_hist / (single_xy + 1e-10) + 1e-10).log2()).sum()
        value = max(0, value.item())
        res.append(value)
    return i, res


def get_train_data(args):
    feature_file = os.path.join(args.feature_prefix, 'feature.pt')
    feature = t.load(feature_file, map_location='cpu')
    book_file = os.path.join(args.book_prefix, 'book_{}.pt'.format(args.num_bit))
    book = t.load(book_file, map_location='cpu')
    book = {k: v.float() for k, v in book.items() if k in ['rB', 'rL', 'qB', 'qL']}
    train_size = book['rL'].size(0)
    train_idx = t.as_tensor(random.sample(range(train_size), args.num_train), dtype=t.int64)

    train_feature = feature['rF'][train_idx].to(args.device)
    train_book = book['rB'][train_idx].clamp(0, 1).to(args.device)
    train_label = book['rL'][train_idx].to(args.device)

    # TODO: NEED THIS?
    # normalize
    train_feature_mean = train_feature.mean(dim=0, keepdim=True)
    # train_feature_std = train_feature.std(dim=0, keepdim=True)
    # train_feature = (train_feature - train_feature_mean) / (train_feature_std + 1e-6)
    train_feature = train_feature - train_feature_mean
    

    return train_feature, train_book, train_label


def train(args):
    num_bit = args.num_bit
    num_train = args.num_train
    device = args.device

    train_feature, train_book, train_label = get_train_data(args)

    # get distance mat
    train_feature_2 = (train_feature**2).sum(dim=1, keepdim=True)
    train_dist_mat = train_feature_2 + train_feature_2.t() - 2 * train_feature @ train_feature.t()
    train_dist_mat.clamp_min_(0)  # TR,TR
    train_ind_mat = (train_label @ train_label.t()) > 0

    selected_train_dist = train_dist_mat.where(train_ind_mat, t.full_like(train_dist_mat, np.inf))
    selected_idx = selected_train_dist.argsort(dim=1)[:, 1:1 + args.num_same_neighbor]
    selected_ind = t.zeros_like(train_ind_mat).scatter(1, selected_idx, True)
    selected_ind &= train_ind_mat

    s_mat = t.zeros((num_train, num_train), dtype=t.float32, device=device)  # TR,TR
    s_mat[selected_ind] = train_dist_mat[selected_ind]
    s_mat = (s_mat + s_mat.t()) / 2
    s_mat = (s_mat > 0) * (-s_mat / s_mat.max()).exp()

    # get points
    s_mat_sum = s_mat.sum(dim=0, keepdim=True)**-0.5
    normed_s_mat = s_mat * s_mat_sum * s_mat_sum.t()
    L_mat = t.eye(num_train, dtype=t.float32, device=device) - normed_s_mat
    points = (train_book.t() @ L_mat @ train_book).diag()

    # get edges
    edges = t.zeros((num_bit, num_bit), dtype=t.float32, device=device)
    if args.num_worker > 0:
        # os.system('ulimit -n 102400')
        pool = mp.Pool(processes=args.num_worker)
        results = []
        for i in range(num_bit):
            results.append(pool.apply_async(compute_edge, (i, train_book.cpu(), num_bit)))
        pool.close()
        # print('closed')
        pool.join()
        # print('join')
        for res in results:
            i, values = res.get()
            for j in range(i + 1, num_bit):
                edges[i, j] = values[j - i - 1]
        # print('mp over')
    else:
        # Old Method
        temp_ind = t.ones(5, dtype=t.bool, device=device)
        temp_ind[2] = False
        for i in range(num_bit):
            for j in range(i + 1, num_bit):
                bits_i = train_book[:, i]
                bits_j = train_book[:, j]
                x_hist = bits_i.histc(bins=2, min=0, max=1) / num_train
                y_hist = bits_j.histc(bins=2, min=0, max=1) / num_train
                bits_ij = (bits_i + bits_j) * 2 - (bits_i - bits_j)
                res_hist = bits_ij.histc(bins=5, min=0, max=4) / num_train
                xy_hist = t.stack((t.stack((res_hist[0], res_hist[1] + res_hist[2])), res_hist[3:])).t()
                single_xy = x_hist.unsqueeze(1) * y_hist.unsqueeze(0)
                edges[i, j] = (xy_hist * (xy_hist / (single_xy + 1e-10) + 1e-10).log2()).sum()

    graph = {'points': points.cpu(), 'edges': edges.cpu()}
    graph_file = os.path.join(args.exp_path, 'graph.pt')
    t.save(graph, graph_file)
    # print('Over')
