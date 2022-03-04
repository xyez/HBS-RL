import scipy.signal
import torch as t


def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, -discount], x[::-1], axis=0)[::-1]


class Buffer():
    def __init__(self, obs_dim, act_dim, size, num_envs, gamma=0.99, lam=0.95, device='cpu'):
        self.obs_buf = t.zeros((size, num_envs, obs_dim), dtype=t.float32, device=device)
        self.act_buf = t.zeros((size, num_envs), dtype=t.int64, device=device)
        self.logp_buf = t.zeros((size, num_envs), dtype=t.float32, device=device)
        self.rew_buf = t.zeros((size, num_envs), dtype=t.float32, device=device)
        self.remain_buf = t.zeros((size, num_envs), dtype=t.float32, device=device)

        self.val_buf = t.zeros((size + 1, num_envs), dtype=t.float32, device=device)

        self.adv_buf = t.zeros((size, num_envs), dtype=t.float32, device=device)
        self.ret_buf = t.zeros((size, num_envs), dtype=t.float32, device=device)

        self.gamma = gamma
        self.lam = lam
        self.device = t.device(device)
        self.start_idx, self.ptr, self.max_size = 0, 0, size
        self.num_envs = num_envs
        self.default_last_val = t.zeros((1, num_envs), dtype=t.float32, device=device)

    def add(self, obs, act, rew, done, val, logp):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act.squeeze(1)
        self.rew_buf[self.ptr] = rew
        self.remain_buf[self.ptr] = 1 - done.float()
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def get(self):
        assert self.ptr == self.max_size
        self.start_idx, self.ptr = 0, 0

        advs = (self.adv_buf - self.adv_buf.mean()) / self.adv_buf.std()  # PA,NE
        return self.obs_buf, self.act_buf, self.logp_buf, advs, self.ret_buf, self.val_buf

    def finish_path_0(self, last_val=0):
        if last_val == 0:
            last_val = self.default_last_val
        path_slice = slice(self.start_idx, self.ptr)
        rews = t.cat((self.rew_buf[path_slice], last_val), dim=0)
        vals = t.cat((self.val_buf[path_slice], last_val), dim=0)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = t.as_tensor(discount_cumsum(deltas.cpu().numpy(), self.gamma * self.lam).copy(), dtype=t.float32, device=self.device)
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.start_idx = self.ptr

    def finish_buffer(self, last_val):
        self.val_buf[self.ptr] = last_val
        deltas = self.rew_buf + self.gamma * self.remain_buf * self.val_buf[1:] - self.val_buf[:-1]
        return_i = last_val
        adv_i = 0
        for i in reversed(range(self.max_size)):
            return_i = self.rew_buf[i] + self.gamma * self.remain_buf[i] * return_i
            self.ret_buf[i] = return_i

            adv_i = deltas[i] + self.gamma * self.lam * self.remain_buf[i] * adv_i
            self.adv_buf[i] = adv_i
