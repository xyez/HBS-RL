class Candidate(object):
    def __init__(self, num_candi=8, num_val_times=4):
        self._num_candi = num_candi
        self._num_val_times = num_val_times
        self._state = None
        self._map_train = 0
        self._map_test = 0

    def update(self, state_add, map_train_add, map_test_add, compute_map):
        idx_add = map_train_add.argsort(0, descending=True)[:self._num_candi]
        state_add = state_add[idx_add]  # num_candi,num_bit
        map_train_add = map_train_add[idx_add]  # num_candi
        map_test_add = map_test_add[idx_add]  # num_candi

        map_train, map_test = compute_map(state_add, num_times=self._num_val_times - 1)
        map_train = (map_train_add + (self._num_val_times - 1) * map_train) / self._num_val_times
        map_test = (map_test_add + (self._num_val_times - 1) * map_test) / self._num_val_times
        map_idx = map_train.argmax()
        map_max = map_train[map_idx].item()
        if map_train[map_idx] > self._map_train:
            self._map_train = map_max
            self._map_test = map_test[map_idx].item()
            self._state = state_add[map_idx]

    def get_map_test(self):
        return self._map_test

    def get_map_train(self):
        return self._map_train

    def get_state_dict(self):
        state_dict = {
            'state': self._state,
            'map_train': self._map_train,
            'map_test': self._map_test
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self._state = state_dict['state']
        self._map_train = state_dict['map_train']
        self._map_test = state_dict['map_test']
