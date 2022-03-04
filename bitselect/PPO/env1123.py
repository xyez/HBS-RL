import os
import random

import numpy as np
import torch as t

from bitselect.PPO.gym_classes import Box, Discrete


class BitSelectEnv():
    def __init__(self, num_env, num_bit, num_subbit, file_dir, device, reward_step=0, use_test=False, is_train=True, subref_size=0, subval_size=0, subtest_size=0, trainset_size=0, testset_size=0):
        self.num_env = num_env
        self.num_bit = num_bit
        self.num_subbit = num_subbit
        self.file_dir = file_dir
        self.device = device
        self.use_test = use_test
        self.is_train = is_train

        self.subref_size = subref_size
        self.subval_size = subval_size
        self.subtest_size = subtest_size
        self.trainset_size = trainset_size
        self.testset_size = testset_size

        self.observation_space = Box(low=-1., high=1., shape=(self.num_bit,), dtype=np.float32)
        self.action_space = Discrete(self.num_bit)
        self.default_reward = t.zeros(self.num_env, dtype=t.float32, device=self.device)
        self.default_info = {}

        self.state = t.zeros((self.num_env, self.num_bit), dtype=t.bool, device=self.device)  # NE,S
        self._step = 0
        self._reward_step = reward_step if reward_step > 0 else num_subbit

        if self.is_train:
            self._init_data(file_dir)

    def _init_data(self, file_dir):
        data = t.load(os.path.join(file_dir, 'book_{}.pt'.format(self.num_bit)), map_location=self.device)

        rB = data['rB'].float()  # 69000,500
        rL = data['rL'].float()
        qB = data['qB'].float()  # 69000,500
        qL = data['qL'].float()

        if self.trainset_size <= 0:
            self.trainset = rB
            self.trainset_label = rL
        else:
            trainset_idx = t.as_tensor(random.sample(range(rB.size(0)), self.trainset_size), device=self.device)
            self.trainset = rB[trainset_idx]
            self.trainset_label = rL[trainset_idx]
        if self.testset_size <= 0:
            self.testset = qB
        else:
            testset_idx = t.as_tensor(random.sample(range(qB.size(0)), self.testset_size), device=self.device)
            self.testset = qB[testset_idx]
            self.testset_label = qL[testset_idx]

        self.subref_range = t.arange(1, 1 + self.subref_size).to(self.device).float().unsqueeze(0)

        self.minibatch_size = 8
        # self.batch_arange = t.arange(self.num_subbit, device=self.device).unsqueeze(0).repeat(self.num_env, 1)

        self.use_h_mat = True
        ndomset_dir = os.path.join(self.file_dir, 'NDomSet')
        if os.path.exists(ndomset_dir):
            self.h_mat = t.load(os.path.join(ndomset_dir, 'checkpoint-{}-{}.pt'.format(self.num_bit, self.num_subbit)), map_location=self.device)['H_mat']
            print(os.path.join(ndomset_dir.replace('datasets', 'results', ), 'NDomSet_idx-{}-{}.npy'.format(self.num_bit, self.num_subbit)))
            self.NDomSet_idx = t.as_tensor(np.load(os.path.join(ndomset_dir.replace('datasets', 'results', ), 'NDomSet_idx-{}-{}.npy'.format(self.num_bit, self.num_subbit))), dtype=t.int64, device=self.device)
            self.baseline_NDomSet_value = self.h_mat[self.NDomSet_idx, :][:, self.NDomSet_idx].sum()
            self.Random_idx = t.as_tensor(random.sample(range(self.num_bit), self.num_subbit)).to(self.device)
            self.baseline_Random_value = self.h_mat[self.Random_idx, :][:, self.Random_idx].sum()
        else:
            self.use_h_mat = False
            self.baseline_Random_value = 0
            self.baseline_NDomSet_value = 0
        self.default_info = {'Random': self.baseline_Random_value,
                             'NDomSet': self.baseline_NDomSet_value,
                             }

    def _get_idxs(self, num_times):
        subrB_idx = t.as_tensor(random.sample(range(self.trainset.size(0)), self.subref_size + self.subval_size * num_times), device=self.device)
        subref_idx = subrB_idx[:self.subref_size]
        subval_idx = subrB_idx[self.subref_size:]
        subtest_idx = t.as_tensor(random.sample(range(self.testset.size(0)), self.subtest_size), device=self.device)
        return subref_idx, subval_idx, subtest_idx

    def reset(self):
        self._step = 0
        self.state.zero_()
        return self.state.float()

    def step(self, action):
        # shape
        # action: NE,1
        self._step += 1
        self.state.scatter_(1, action, 1)
        done = self.state.sum(1) == self.num_subbit
        if self._step % self._reward_step == 0 and self.is_train:
            if self.use_h_mat:
                self.values = self._get_value(self.state.float())  # NE
            else:
                self.values = t.zeros(self.num_env)
            maps = self.get_maps(self.state)  # NE
            reward = maps[1] if self.use_test else maps[0]
            # reward=0.05*xax+20*map
            multi = {'EpRet': reward.cpu().numpy(),
                     'RL': self.values.cpu().numpy(),
                     'MAP_TRAIN': maps[0].cpu().numpy(),
                     'MAP_TEST': maps[1].cpu().numpy(),
                     }
            info = {'single': self.default_info,
                    'multi': multi}
        else:
            info = {'single': self.default_info,
                    'multi': {}}
            reward = self.default_reward

        if self.is_train:
            last_ind = self.state.sum(1) == self.num_subbit
            if last_ind.any().item():
                last_state = self.state[last_ind]
                info['last'] = {'episode': last_ind.sum().item(),
                                'state': last_state,
                                'map_train': maps[0],
                                'map_test': maps[1]}
                self.state[last_ind] = 0
        return self.state.float(), reward, done, info

    def _get_value(self, inds):
        values = t.as_tensor([inds[i] @ self.h_mat @ inds[i] for i in range(inds.size(0))], dtype=t.float32).to(self.device)
        return values

    def get_maps(self, state, num_times=1):
        subref_idx, query_idx_train, query_idx_test = self._get_idxs(num_times)

        refbook = self.trainset[subref_idx]  # TRAIN,ALL
        refbook_label = self.trainset_label[subref_idx]
        querybook_train = self.trainset[query_idx_train]  # TRAIN,ALL
        querybook_label_train = self.trainset_label[query_idx_train]
        m_ap_train = self._compute_mAP(state, refbook, refbook_label, querybook_train, querybook_label_train, num_interval=100)

        querybook_test = self.testset[query_idx_test]
        querybook_label_test = self.testset_label[query_idx_test]
        m_ap_test = self._compute_mAP(state, refbook, refbook_label, querybook_test, querybook_label_test, num_interval=100)

        return [m_ap_train, m_ap_test]

    def _compute_mAP(self, state, trainbook, trainlabel, testbook, testlabel, num_interval=10):
        num_state = state.size(0)
        batch_arange = t.arange(self.num_subbit, device=self.device).unsqueeze(0).repeat(num_state, 1)
        num_minibatch = int(np.ceil(num_state / self.minibatch_size))
        batch_ind = batch_arange < state.sum(1, keepdim=True)
        batch_subrefbook = t.zeros((num_state, self.num_subbit, self.subref_size), dtype=t.float32, device=self.device)  # NE,SUB,TRAIN
        for i in range(num_minibatch):
            idx = i * self.minibatch_size
            idx_next = min(num_state, (i + 1) * self.minibatch_size)
            idx_slice = slice(idx, idx_next)
            minibatch_size = idx_next - idx
            minibatch_refbook_shape = (minibatch_size, self.num_bit, self.subref_size)
            batch_subrefbook_shape = (minibatch_size, self.num_subbit, self.subref_size)
            batch_subrefbook[idx_slice][batch_ind[idx_slice].unsqueeze(-1).expand(batch_subrefbook_shape)] = trainbook.t().expand(minibatch_refbook_shape)[state[idx_slice].unsqueeze(-1).expand(minibatch_refbook_shape)]
        batch_subtestbook_shape = (num_state, self.num_subbit, testbook.size(0))  # NE,SUB,TEST
        batch_testbook_shape = (num_state, self.num_bit, testbook.size(0))
        batch_testbook = t.zeros(batch_subtestbook_shape, dtype=t.float32, device=self.device)
        batch_testbook[batch_ind.unsqueeze(-1).expand(batch_subtestbook_shape)] = testbook.t().expand(batch_testbook_shape)[state.unsqueeze(-1).expand(batch_testbook_shape)]
        batch_testbook = batch_testbook.permute(0, 2, 1)  # NE,NSAMPLE,SUB
        ap = t.zeros((num_state, testbook.size(0)), dtype=t.float32, device=self.device)
        for i in range(0, testbook.size(0), num_interval):
            # testbook_i = testbook[i:i + num_interval]  # TEST,ALL
            testlabel_i = testlabel[i:i + num_interval]
            groundtruth = (testlabel_i @ trainlabel.t()).clamp_max(1)  # TEST,TRAIN
            # testbook_i = testbook_i.t()[self.state_idx].permute(0, 2, 1)  # NE,TEST,SUB
            testbook_i = batch_testbook[:, i:i + num_interval, :]
            groundtruth_sum = groundtruth.sum(dim=1).unsqueeze(0)  # 1,TEST
            groundtruth = groundtruth.unsqueeze(0).repeat(num_state, 1, 1)  # NE,TEST,TRAIN
            distance = 0.5 * (self.num_subbit - testbook_i.bmm(batch_subrefbook))  # NE,TEST,TRAIN
            distance_idx = distance.argsort(dim=2)  # NE,TEST,TRAIN
            groundtruth_sort = groundtruth.gather(dim=2, index=distance_idx)  # NE,TEST,TRAIN
            rank_groundtruth = groundtruth_sort.cumsum(dim=2)  # NE,TEST,TRAIN
            # ind_valud = (groundtruth_sum > 0).float()  # 1,TEST
            ap_i = (groundtruth_sort * (rank_groundtruth / self.subref_range)).sum(dim=2) / groundtruth_sum.clamp_min_(1e-8)
            ap[:, i:i + num_interval] = ap_i
            # t.cuda.empty_cache()
        m_ap = ap.mean(dim=1)
        return m_ap
