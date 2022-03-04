import numpy as np
from torch.autograd import Variable


def compress(train, test, model, classes=10):
    retrievalB = list([])
    retrievalL = list([])
    for batch_step, (data, target) in enumerate(train):
        var_data = Variable(data.cuda())
        _, _, code = model(var_data)
        retrievalB.extend(code.cpu().data.numpy())
        retrievalL.extend(target)

    queryB = list([])
    queryL = list([])
    for batch_step, (data, target) in enumerate(test):
        var_data = Variable(data.cuda())
        _, _, code = model(var_data)
        queryB.extend(code.cpu().data.numpy())
        queryL.extend(target)

    retrievalB = np.array(retrievalB)
    retrievalL = np.eye(classes)[np.array(retrievalL)]

    queryB = np.array(queryB)
    queryL = np.eye(classes)[np.array(queryL)]
    return retrievalB, retrievalL, queryB, queryL


def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    q = B2.shape[1]  # max inner product value
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def calculate_map(qB, rB, queryL, retrievalL):
    """
       :param qB: {-1,+1}^{mxq} query bits
       :param rB: {-1,+1}^{nxq} retrieval bits
       :param queryL: {0,1}^{mxl} query label
       :param retrievalL: {0,1}^{nxl} retrieval label
       :return:
    """
    num_query = queryL.shape[0]
    map = 0
    for iter in range(num_query):
        # gnd : check if exists any retrieval items with same label
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        # tsum number of items with same label
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        # sort gnd by hamming dist
        hamm = calculate_hamming(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        count = np.linspace(1, tsum, tsum)  # [1,2, tsum]
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        # print(map_)
        map = map + map_
    map = map / num_query
    return map


def calculate_top_map(qB, rB, queryL, retrievalL, topk):
    """
    :param qB: {-1,+1}^{mxq} query bits
    :param rB: {-1,+1}^{nxq} retrieval bits
    :param queryL: {0,1}^{mxl} query label
    :param retrievalL: {0,1}^{nxl} retrieval label
    :param topk:
    :return:
    """
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = calculate_hamming(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = int(np.sum(tgnd))
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        # print(topkmap_)
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap


import torch as t


def compute_mAP(testbook, codebook, groundtruth=None, testlabel=None, codelabel=None, topK=None, num_interval=20):
    if groundtruth is None:
        groundtruth = testlabel @ codelabel.t()
        groundtruth.clamp_max_(1.)
    if topK is None:
        topK = codebook.size(0)
    device = codebook.device
    num_test = testbook.size(0)
    train_range = t.arange(1, 1 + topK).to(device).float().unsqueeze(0)
    # testbook2_sum = testbook.pow(2).sum(dim=1, keepdim=True)
    # codebook2_sum = codebook.pow(2).sum(dim=1, keepdim=True)

    recall = t.zeros(topK, device=device)
    precision = t.zeros(topK, device=device)
    ap = t.zeros(num_test, device=device)
    for i in range(0, testbook.size(0), num_interval):
        if i + num_interval > testbook.size(0):
            end = testbook.size(0)
        else:
            end = i + num_interval
        #   distance = testbook2_sum[i:i + num_interval] - 2 * testbook[i:i + num_interval] @ codebook.t() + codebook2_sum.t()
        distance = testbook.size(1) - testbook[i:end] @ codebook.t()
        distance_idx = distance.argsort(dim=1)[:, :topK]
        groundtruth_sort = groundtruth[i:end].gather(dim=1, index=distance_idx)  # TEST,topK
        groundtruth_sort_sum = groundtruth_sort.sum(dim=1, keepdim=True)  # TEST,1
        groundtruth_sort_cumsum = groundtruth_sort.cumsum(dim=1)  # TEST,topK
        ind_valud = (groundtruth_sort_sum > 0).float()  # TEST,1
        # groundtruth_sort_sum.clamp_min_(1e-12)

        # recall += ((groundtruth_sort_cumsum / groundtruth_sort_sum)).sum(0)
        # print('groundtruth_sort_cumsum',groundtruth_sort_cumsum.shape)
        # print((ind_valud * (groundtruth_sort_cumsum / groundtruth_sort_sum)).sum(0).shape,'0')
        recall += (ind_valud * (groundtruth_sort_cumsum / groundtruth_sort_sum)).sum(0)
        precision_i = groundtruth_sort_cumsum / train_range
        # print(precision_i.shape,'1')
        precision += (ind_valud * precision_i).sum(0)
        ap[i:end] = (ind_valud * ((groundtruth_sort * precision_i).sum(dim=1, keepdim=True) / groundtruth_sort_sum)).squeeze(1)
    recall /= num_test
    precision /= num_test
    m_ap = ap.mean()
    # print(recall.cpu().data.numpy(), precision.cpu().data.numpy(), ap.cpu().data.numpy())
    return recall.cpu().numpy(), precision.cpu().numpy(), m_ap.item()


def get_results(books, device, topK=0):
    if topK == 0:
        topK = None

    results = {}
    for book_name, book in books.items():
        codebook = book['rB'].to(device)
        testbook = book['qB'].to(device)
        groundtruth = book['qL'].to(device) @ book['rL'].to(device).t()
        groundtruth.clamp_max_(1.)
        # print(groundtruth.cpu().data.numpy())
        recall, precision, mAP = compute_mAP(testbook, codebook, groundtruth, topK=topK)
        results[book_name] = {'recall': recall,
                              'precision': precision,
                              'map': mAP}
    return results
