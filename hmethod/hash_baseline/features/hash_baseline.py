import argparse
import os

import numpy as np
import torch as t
from sklearn import decomposition


def PCAH(trainX, testX, nbits, pcafile):
    num = trainX.shape[0]
    idx = np.arange(num)
    np.random.shuffle(idx)
    pca = decomposition.PCA()
    pca.fit(trainX[idx[:10000], :])
    pca.n_components = nbits
    trainX = pca.fit_transform(trainX)
    testX = pca.fit_transform(testX)

    np.savez(pcafile, trainX, testX)
    trainX[trainX >= 0] = 1
    trainX[trainX < 0] = -1
    testX[testX >= 0] = 1
    testX[testX < 0] = -1
    return testX, trainX


def ITQ(trainX, testX, nbits, pcafile):
    data = np.load(pcafile)
    trainX = data['arr_0']
    testX = data['arr_1']

    R = np.random.randn(nbits, nbits)
    [U11, S2, V2] = np.linalg.svd(R)
    R = U11[:, :nbits]

    for i in range(50):
        Z = np.dot(trainX, R)
        UX = -np.ones(Z.shape)
        UX[Z >= 0] = 1
        C = np.dot(UX.T, trainX)
        [UB, sigma, UA] = np.linalg.svd(C)
        R = np.dot(UA, UB.T)

    trainX = np.dot(trainX, R)
    testX = np.dot(testX, R)
    trainX[trainX >= 0] = 1
    trainX[trainX < 0] = -1
    testX[testX >= 0] = 1
    testX[testX < 0] = -1
    return testX, trainX


def SH(trainX, testX, nbits, pcafile):
    #   pca = decomposition.PCA()
    #  pca.fit(trainX)
    # pca.n_components = nbits
    # trainX = pca.fit_transform(trainX)
    # testX = pca.fit_transform(testX)
    data = np.load(pcafile)
    trainX = data['arr_0']
    testX = data['arr_1']

    eps = 1e-12
    mn = np.min(trainX, axis=0) - eps
    mx = np.max(trainX, axis=0) + eps

    R = mx - mn
    maxMode = np.ceil((nbits + 1) * R / np.max(R)).astype('int32')
    nModes = int(np.sum(maxMode) - len(maxMode) + 1)
    modes = np.ones((nModes, nbits))
    # print(nModes, maxMode)
    m = 0
    for i in range(nbits):
        modes[m + 1:m + maxMode[i], i] = np.arange(2, maxMode[i] + 1)
        m = m + maxMode[i] - 1
    modes = modes - 1
    pi = 3.1415926
    omega0 = pi / R
    omegas = modes * np.tile(omega0, [nModes, 1])
    eigVal = -np.sum(omegas**2, axis=1)
    ii = np.argsort(-eigVal)
    # print(modes)
    modes = modes[ii[1:nbits + 1], :]
    # print(modes)

    trainX = trainX - np.tile(mn, [trainX.shape[0], 1])
    omegas = modes * np.tile(omega0, [nbits, 1])

    U = np.zeros((trainX.shape[0], nbits))
    for i in range(nbits):
        omegai = np.tile(omegas[i, :], [trainX.shape[0], 1])
        ys = np.sin(trainX * omegai + pi / 2)
        yi = np.prod(ys, axis=1)
        U[:, i] = yi

    testX = testX - np.tile(mn, [testX.shape[0], 1])
    V = np.zeros((testX.shape[0], nbits))
    for i in range(nbits):
        omegai = np.tile(omegas[i, :], [testX.shape[0], 1])
        ys = np.sin(testX * omegai + pi / 2)
        yi = np.prod(ys, axis=1)
        V[:, i] = yi
    # print(V, U)
    return V, U


def SpH(trainX, testX, nbits):
    def random_center(data, nbits):
        [N, D] = data.shape
        centers = np.zeros((nbits, D))
        for i in range(nbits):
            R = np.arange(N)
            np.random.shuffle(R)
            sample = data[R[:5], :]
            centers[i, :] = np.mean(sample, axis=0)
        return centers

    def compute_statistics(data, centers):
        [N, D] = data.shape
        dist = np.reshape(np.sum(centers * centers, axis=1), [-1, 1]) - 2 * np.dot(centers, data.T) + \
               np.reshape(np.sum(data * data, axis=1), [1, -1])
        radii = np.sort(dist, axis=1)
        radii = radii[:, int(N / 2)]
        dist = np.where(dist <= np.tile(np.reshape(radii, [-1, 1]), [1, N]), np.ones(dist.shape), np.zeros(dist.shape))
        O1 = np.sum(dist, axis=1)

        avg = 0
        avg2 = 0
        nbits = centers.shape[0]
        O2 = np.dot(dist, dist.T)
        for i in range(nbits - 1):
            for j in range(i + 1, nbits):
                avg = avg + np.abs(O2[i, j] - N / 4)
                avg2 = avg2 + O2[i, j]

        avg = avg / (nbits * (nbits - 1) / 2)
        avg2 = avg2 / (nbits * (nbits - 1) / 2)
        stddev = 0
        for i in range(nbits - 1):
            for j in range(i + 1, nbits):
                stddev = stddev + ((O2[i, j] - avg2)**2)
        stddev = np.sqrt(stddev / (nbits * (nbits - 1)) / 2)
        return O1, O2, radii, avg, stddev

    centers = random_center(trainX, nbits)
    O1, O2, radii, avg, stddev = compute_statistics(trainX, centers)

    it = 1
    [N, D] = trainX.shape
    while True:
        forces = np.zeros((nbits, D))
        for i in range(nbits - 1):
            for j in range(i, nbits):
                force = (0.5 * (O2[i, j] - N / 4) / (N / 4) * (centers[i, :] - centers[j, :]))
                forces[i, :] = forces[i, :] + force / nbits
                forces[j, :] = forces[j, :] - force / nbits
        centers = centers + forces

        O1, O2, radii, avg, stddev = compute_statistics(trainX, centers)

        if avg < 0.2 * N / 4 and stddev <= 0.15 * N / 4:
            break
        if it >= 100:
            print('iter excedd 100, avg=%f, stddev=%f\n' % {avg, stddev})

        it += 1

    radii = np.reshape(radii, [1, -1])
    dist = np.reshape(np.sum(trainX * trainX, axis=1), [-1, 1]) - 2 * np.dot(trainX, centers.T) + \
           np.reshape(np.sum(centers * centers, axis=1), [1, -1])
    D_code = np.where(dist <= np.tile(radii, [N, 1]), np.ones((N, nbits)), np.zeros((N, nbits)))
    dist = np.reshape(np.sum(testX * testX, axis=1), [-1, 1]) - 2 * np.dot(testX, centers.T) + \
           np.reshape(np.sum(centers * centers, axis=1), [1, -1])
    Q_code = np.where(dist <= np.tile(radii, [testX.shape[0], 1]), np.ones((testX.shape[0], nbits)), np.zeros((testX.shape[0], nbits)))

    return Q_code, D_code


def compute_mAP(Qv, Dv, Q_label, D_label, topK):
    dis = np.reshape(np.sum(Qv * Qv, axis=1), [-1, 1]) - 2 * np.dot(Qv, Dv.T) + \
          np.reshape(np.sum(Dv * Dv, axis=1), [1, -1])
    idx = np.argsort(dis, axis=1)
    mAP = []
    precision = []
    recall = []
    for i in range(Qv.shape[0]):
        ri = idx[i, :]
        match = np.dot(np.reshape(Q_label[i, :], [1, -1]), D_label.T)
        match[match > 1] = 1.
        match = np.array(match[0, ri]).astype('float32')
        nrel = np.sum(match[:topK])
        pre = np.cumsum(match) / (1. + np.arange(idx.shape[1]))
        if nrel > 0:
            rec = np.cumsum(match) / nrel
            mAP.append(np.sum(pre[:topK] * match[:topK]) / np.sum(match[:topK]))
            precision.append(pre)
            recall.append(rec)
        else:
            mAP.append(0)
            precision.append(0)
            recall.append(0)
    precision = np.mean(np.array(precision), axis=0)
    recall = np.mean(np.array(recall), axis=0)
    mAP = np.mean(mAP)
    print('mAP: ', mAP)
    res = {'precision': precision, 'recall': recall, 'map': mAP}
    return res


def main(args):
    feature_file = 'data/feature/{}/feature.pt'.format(args.dataset)
    feature_data = t.load(feature_file, map_location='cpu')
    query = feature_data['qF'].cpu().numpy()
    db = feature_data['rF'].cpu().numpy()
    query_L = feature_data['qL'].cpu().numpy()
    db_L = feature_data['rL'].cpu().numpy()
    book_label = {'qL': feature_data['qL'], 'rL': feature_data['rL']}
    pt_path = os.path.join('data', 'book', args.dataset)
    os.makedirs(pt_path, exist_ok=True)
    pca_path = os.path.join(pt_path, 'PCA')
    os.makedirs(pca_path, exist_ok=True)
    m_path = {}
    for method in args.methods:
        m_path[method] = os.path.join(pt_path, method)
        os.makedirs(m_path[method], exist_ok=True)
    file_map = os.path.join(pt_path, 'result_map.txt')

    result_map = {'SH': [], 'ITQ': [], 'SpH': []}
    for num_bit in args.num_bits:
        print('\nNum Bit: ', num_bit)
        print('PCAH.........')
        pca_file = os.path.join(pca_path, 'pca_{}.npz'.format(num_bit))
        Q_code, D_code = PCAH(db, query, num_bit, pca_file)
        result = compute_mAP(Q_code, D_code, query_L, db_L, args.topK)
        book = {'qB': t.as_tensor(Q_code, dtype=t.float32).sign().clamp_min(0), 'rB': t.as_tensor(D_code, dtype=t.float32).sign().clamp_min(0), 'res': result, **book_label}
        t.save(book, os.path.join(pca_path, 'book_{}.pt'.format(num_bit)))

        for method in args.methods:
            if method == 'SH':
                print('SH.........')
                Q_code, D_code = SH(db, query, num_bit, pca_file)
            elif method == 'ITQ':
                print('ITQ.........')
                Q_code, D_code = ITQ(db, query, num_bit, pca_file)
            elif method == 'SpH':
                print('SpH.........')
                Q_code, D_code = SpH(db, query, num_bit)

            else:
                raise NotImplementedError()
            result = compute_mAP(Q_code, D_code, query_L, db_L, args.topK)
            book = {'qB': t.as_tensor(Q_code, dtype=t.float32).sign().clamp_min(0), 'rB': t.as_tensor(D_code, dtype=t.float32).sign().clamp_min(0), 'res': result, **book_label}
            t.save(book, os.path.join(m_path[method], 'book_{}.pt'.format(num_bit)))
            result_map[method].append(result['map'])
    if args.log:
        with open(file_map, 'w') as f:
            f.write(', '.join(list(map(str, args.num_bits))) + '\n')
            f.write('SH: ' + ', '.join(map(lambda x: '{:.6f}'.format(x), result_map['SH'])) + '\n')
            f.write('ITQ: ' + ', '.join(map(lambda x: '{:.6f}'.format(x), result_map['ITQ'])) + '\n')
            f.write('SpH: ' + ', '.join(map(lambda x: '{:.6f}'.format(x), result_map['SpH'])) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--methods', type=str, nargs='*', default=['SH', 'ITQ', 'SpH'])
    parser.add_argument('--log', type=eval, default=True)
    parser.add_argument('--num_bits', type=int, nargs='*', default=[16, 24, 32, 48, 64, 128, 512])
    parser.add_argument('--topK', type=int, default=5000)
    args = parser.parse_args()
    main(args)
