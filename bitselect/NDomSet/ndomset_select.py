import os

import torch as t


def ndomset_select(graph, num_train, num_sub_bits, num_same_neighbors, gamma, lambda_, base_dir, device, num_workers):
    checkpoint_file = os.path.join(base_dir, 'checkpoint-{}.pt'.format(bitinfo))
    if os.path.exists(checkpoint_file):
        graph = t.load(checkpoint_file, map_location='cpu')
    else:
        graph = get_graph(data, num_train, num_same_neighbors, gamma, lambda_, 'cpu', num_workers)
        t.save(graph, checkpoint_file)
    for k, v in graph.items():
        graph[k] = v.to(device)
    idx = get_bits(graph, num_sub_bits)
    return idx.cpu().numpy()
    # idx = get_bits_matlab(graph, num_sub_bits)
    # return idx
