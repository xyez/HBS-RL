import os

import numpy as np
import torch as t


def rl_select_(file_dir, bitinfo):
    file = os.path.join(file_dir, 'checkpoint-{}.pt'.format(bitinfo))
    print(file_dir)
    print('*' * 100)
    rlidxs = t.load(file, map_location='cpu')
    sub_idx = np.where(rlidxs['idx'][-2] > 0)[0]
    print(len(sub_idx))
    return sub_idx


def rl_select(file_dir, bitinfo):
    from bitselect.PPO.agent import Agent
    from bitselect.PPO.env1123 import BitSelectEnv
    file = os.path.join(file_dir, 'checkpoint-{}.pt'.format(bitinfo))
    rlidxs = t.load(file, map_location='cpu')
    sub_ind = (rlidxs['idx'][-2] > 0).numpy()
    num_bits = len(sub_ind)
    num_sub_bits = np.sum(sub_ind)
    device = 'cpu'
    env = BitSelectEnv(num_env=1, num_bit=num_bits, num_subbit=num_sub_bits, file_dir='/tmp', device=device, is_train=False)
    base_dir = os.path.join('rl_select/data', *(file_dir.split('/')[1:3]))
    bitinfo = '{}-{}'.format(num_bits, num_sub_bits)
    for f in os.listdir(base_dir):
        if f.startswith(bitinfo):
            base_dir = os.path.join(base_dir, f)
            break
    file = list(os.walk(base_dir))[-1]
    model_file = os.path.join(file[0], file[-1][0])

    agent_temp = t.load(model_file, map_location='cpu')
    print("model file", model_file)

    obs_dim = num_bits
    act_dim = num_bits
    lr = 1e-3
    agent = Agent(obs_dim, act_dim, lr, device, is_train=False)
    agent.pi.load_state_dict(agent_temp.pi.state_dict())
    state = env.reset()
    sub_idx = []
    while True:
        action, logp, v = agent.act(state)
        state, rewards, done, info = env.step(action)
        sub_idx.append(action.item())
        if done:
            break
    return np.asarray(sub_idx)
