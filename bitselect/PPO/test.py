import os

import torch as t

from bitselect.PPO.agent import Agent
from bitselect.PPO.env1123 import BitSelectEnv


def test(args, ind_file):
    state_dict_file = os.path.join(args.exp_path, 'pyt_save/state_dict.pt', )
    # print(state_dict_file)
    state_dict = t.load(state_dict_file, map_location='cpu')

    if args.use_candi:
        # idx = state_dict['state']['map_train'][0].argmax()
        # obsv = state_dict['state']['state'][0, idx].cpu().bool()
        obsv = state_dict['candi']['state'].cpu().bool()
    else:
        obs_dim = args.num_bit
        act_dim = args.num_bit
        agent = Agent(obs_dim, act_dim, args.lr, args.device, is_train=False)
        if 'agent' in state_dict.keys():
            state_dict = state_dict['agent']
        agent.load_param(state_dict)
        env = BitSelectEnv(num_env=1, num_bit=args.num_bit, num_subbit=args.num_subbit, file_dir='', device=args.device, is_train=False)
        obsv = env.reset()
        while True:
            action = agent.get_action(obsv)
            obsv, reward, done, info = env.step(action)
            if done.item():
                break
        obsv = obsv[0]
        obsv = obsv.cpu().bool()
    # print('PPO\n', t.where(obsv)[0])
    t.save(obsv, ind_file)
    # print('PPO Over')
