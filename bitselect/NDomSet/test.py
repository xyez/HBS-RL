import os

import numpy as np
import torch as t


def replicator(row_idx, col_idx, values, num_bits, mini_change=1e-5):
    device=values.device
    compute_value=lambda m,v,i,j:(m[i]*m[j]*v).sum()

    values = values.double().cpu().numpy()
    maxima = np.zeros(num_bits) + 1 / num_bits
    deri = np.zeros(num_bits)
    row_idx=row_idx.cpu().numpy()
    col_idx=col_idx.cpu().numpy()
    pre_value = 1e-8
    while True:
        deri[:]=0
        for i in range(num_bits):
            deri[row_idx[i]]+=maxima[col_idx[i]]*values[i]
            deri[col_idx[i]]+=maxima[row_idx[i]]*values[i]
        
        maxima *= deri
        
        value=np.sum(maxima)
        # value=compute_value(maxima, values, row_idx, col_idx)
        maxima /= np.sum(maxima)
           
        ratio = (value - pre_value) / pre_value
        if ratio < mini_change:
            break
        pre_value = value
    return t.as_tensor(maxima,device=device)


def get_bits_matlab(graph, num_subbit):
    import matlab.engine
    engine = matlab.engine.start_matlab()
    engine.addpath('matlab_code')
    num_bits = graph.size(0)
    H_mat = graph.cpu().numpy()

    bits_selected = np.zeros(0, dtype=np.int)
    bits_remain = np.arange(num_bits)
    while bits_selected.size < num_subbit:
        H_remain_mat = H_mat[bits_remain, :][:, bits_remain]
        Hr_idx_row, Hr_idx_col = np.where(H_remain_mat > 0)
        A_mat = np.stack((Hr_idx_row + 1, Hr_idx_col + 1, H_remain_mat[Hr_idx_row, Hr_idx_col]), axis=1)  # N,3
        A_mat = A_mat[Hr_idx_row < Hr_idx_col]
        ret_dict = engine.triarea(matlab.double(A_mat.tolist()), matlab.double(np.array([[bits_remain.size]]).tolist()))
        maxima = np.squeeze(np.asarray(ret_dict['maxima']))

        sorted_idx = np.argsort(maxima)[::-1]
        pos_sorted_idx = sorted_idx[maxima[sorted_idx] > 0]
        bits_selected = np.concatenate((bits_selected, bits_remain[pos_sorted_idx]))
        bits_remain = np.delete(bits_remain, pos_sorted_idx)
        print(bits_selected.size)
    sub_idx = bits_selected[:num_subbit]
    return sub_idx


def get_bits(graph, num_subbit):
    num_bits = graph.size(0)
    H_mat = graph

    device = H_mat.device
    bits_selected = t.zeros(0, dtype=t.int64, device=device)
    bits_remain = t.arange(num_bits, device=device)
    num_step = 0
    while len(bits_selected) < num_subbit:
        H_remain_mat = H_mat[bits_remain, :][:, bits_remain]
        Hr_idx_row, Hr_idx_col = t.where(H_remain_mat > 0)
        ind = Hr_idx_row < Hr_idx_col
        Hr_idx_row = Hr_idx_row[ind]
        Hr_idx_col = Hr_idx_col[ind]
        Hr_values = H_remain_mat[Hr_idx_row, Hr_idx_col]

        maxima = replicator(Hr_idx_row, Hr_idx_col, Hr_values, len(bits_remain))

        sorted_idx = maxima.argsort(descending=True)
        pos_sorted_idx = sorted_idx[maxima[sorted_idx] > 0.01]
        # print(pos_sorted_idx.shape)
        bits_selected = t.cat((bits_selected, bits_remain[pos_sorted_idx]))
        bits_remain = bits_remain[t.ones_like(bits_remain, dtype=t.bool).index_fill_(0, pos_sorted_idx, False)]
        # print(len(bits_remain))

        # print(num_step)
        # num_step += 1
    sub_idx = bits_selected[:num_subbit]
    # print(sorted(sub_idx.tolist()))
    return sub_idx.cpu()

def get_graph(args):
    graph_file = os.path.join(args.exp_path, 'graph.pt')
    graph = t.load(graph_file, map_location='cpu')
    
    device=t.device(args.device)
    points=graph['points'].to(device)
    edges=graph['edges'].to(device)
    # refine points
    min_points = points.min()
    max_points = points.max()
    points = (points - min_points) / (max_points - min_points)
    points = t.exp(-args.gamma * points)
    # refine edges
    edges.clamp_min_(0)
    edges += edges.t()
    the_eye = t.eye(edges.size(0), dtype=t.float32, device=device)
    min_edges_plus_eye = (edges + the_eye).min()
    max_edges = edges.max()
    edges = (edges - min_edges_plus_eye + the_eye * min_edges_plus_eye) / (max_edges - min_edges_plus_eye)
    edges = t.exp(-args.lambda0 * edges) - the_eye
    edges.clamp_min_(0)
    # compute H mat
    graph = points.unsqueeze(1) * points.unsqueeze(0) * edges
    return graph
    
def test(args, ind_file):
    graph=get_graph(args)
    if args.use_matlab:
        index = get_bits_matlab(graph, args.num_subbit)
    else:
        index = get_bits(graph, args.num_subbit)
    index = t.as_tensor(index, dtype=t.int64)
    index = t.zeros(args.num_bit, dtype=t.bool).scatter(0, index, 1)
    # print('NDomSet\n', t.where(index)[0])
    t.save(index, ind_file)
    # print('NDomSet Over')
