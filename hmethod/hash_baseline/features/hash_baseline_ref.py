import os
import numpy as np
import torch as t
from copy import deepcopy
from sklearn import decomposition


def PCAH(trainX, testX, nbits):
    num = trainX.shape[0]
    idx = np.arange(num)
    np.random.shuffle(idx)
    pca = decomposition.PCA()
    pca.fit(trainX[idx[:20000], :])
    # pca.fit(trainX[idx, :])
    pca.n_components = nbits
    trainX = pca.fit_transform(trainX)
    testX = pca.fit_transform(testX)

    np.savez('results/pca_feat_{}.npz'.format(nbits), trainX, testX)
    trainX[trainX >= 0] = 1
    trainX[trainX < 0] = -1
    testX[testX >= 0] = 1
    testX[testX < 0] = -1
    return testX, trainX


def ITQ(trainX, testX, nbits):
    data = np.load('results/pca_feat_{}.npz'.format(nbits))
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


def SH(trainX, testX, nbits):
    #   pca = decomposition.PCA()
    #  pca.fit(trainX)
    # pca.n_components = nbits
    #trainX = pca.fit_transform(trainX)
    #testX = pca.fit_transform(testX)
    data = np.load('results/pca_feat_{}.npz'.format(nbits))
    trainX = data['arr_0']
    testX = data['arr_1']

    eps = 1e-12
    mn = np.min(trainX, axis=0) - eps
    mx = np.max(trainX, axis=0) + eps

    R = mx - mn
    maxMode = np.ceil((nbits + 1) * R / np.max(R)).astype('int32')
    nModes = int(np.sum(maxMode) - len(maxMode) + 1)
    modes = np.ones((nModes, nbits))
    #print(nModes, maxMode)
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
    #print(modes)
    modes = modes[ii[1:nbits + 1], :]
    #print(modes)

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
    U[U >= 0] = 1
    U[U < 0] = -1
    V[V >= 0] = 1
    V[V < 0] = -1
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
        dist = np.reshape(np.sum(centers*centers, axis=1), [-1, 1]) - 2*np.dot(centers, data.T) +\
                np.reshape(np.sum(data*data, axis=1), [1, -1])
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
    dist = np.reshape(np.sum(trainX*trainX, axis=1), [-1, 1]) - 2*np.dot(trainX, centers.T) +\
                np.reshape(np.sum(centers*centers, axis=1), [1, -1])
    D_code = np.where(dist <= np.tile(radii, [N, 1]), np.ones((N, nbits)), np.zeros((N, nbits))-1)
    dist = np.reshape(np.sum(testX*testX, axis=1), [-1, 1]) - 2*np.dot(testX, centers.T) +\
                np.reshape(np.sum(centers*centers, axis=1), [1, -1])
    Q_code = np.where(dist <= np.tile(radii, [testX.shape[0], 1]), np.ones((testX.shape[0], nbits)), np.zeros((testX.shape[0], nbits))-1)

    # print(Q_code, D_code)
    return Q_code, D_code


def compute_mAP(Qv, Dv, Q_label, D_label, topK, name):
    dis = np.reshape(np.sum(Qv*Qv, axis=1), [-1, 1]) - 2*np.dot(Qv, Dv.T) +\
            np.reshape(np.sum(Dv*Dv, axis=1), [1, -1])
    idx = np.argsort(dis, axis=1)
    mAP = []
    precision = []
    recall = []
    for i in range(Qv.shape[0]):
        ri = idx[i, :]
        match = np.dot(np.reshape(Q_label[i, :], [1, -1]), D_label.T)
        match[match > 1] = 1.
        match = np.array(match[0, ri]).astype('float32')
        nrel = np.sum(match)
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
    info = {'precision': precision, 'recall': recall, 'mAP': mAP}
    return info


if __name__ == "__main__":
    # data = np.load('./alexnet_pretrained_nus_4096.npz')
    # query, db, query_L, db_L = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    data = t.load('data/cifar10.pt', map_location='cpu')
    query, db, query_L, db_L = data['qF'].numpy(), data['rF'].numpy(), data['qL'].numpy(), data['rL'].numpy()

    topK1 = 50000

    # hashing_methods = {'SH': SH, 'ITQ': ITQ, 'SpH': SpH}
    hashing_methods = {'SpH': SpH}
    num_bits = [16,24,32, 48, 64, 128]
    for num_bit in num_bits:
        print('PCAH...{}.........'.format(num_bit))
        Q_code, D_code = PCAH(db, query, num_bit)
        compute_mAP(Q_code, D_code, query_L, db_L, topK1, 'PCAH')
        for hash_name, hash_fun in hashing_methods.items():
            print('{}.........'.format(hash_name))
            Q_code, D_code = hash_fun(db, query, num_bit)
            info = compute_mAP(Q_code, D_code, query_L, db_L, topK1, 'SH')
            book = {'qB': t.as_tensor(deepcopy(Q_code), dtype=t.float32).clamp(min=0, max=1).type(t.uint8), 'rB': t.as_tensor(deepcopy(D_code), dtype=t.float32).clamp(min=0, max=1).type(t.uint8), 'qL': data['qL'], 'rL': data['rL'], 'info': info}
            base_dir = 'results/book/{}/{}'.format(hash_name, num_bit)
            os.makedirs(base_dir, exist_ok=True)
            t.save(book, os.path.join(base_dir, 'book_{}.pt'.format(num_bit)))
