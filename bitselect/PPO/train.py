import time

from bitselect.PPO.agent import Agent
from bitselect.PPO.buffer import Buffer
from bitselect.PPO.candidate import Candidate
from bitselect.PPO.env1123 import BitSelectEnv
from bitselect.PPO.logx import EpochLogger, get_saved_config


def train(args):
    logger = EpochLogger(output_dir=args.exp_path, exp_name=args.exp_name)
    logger.save_config(get_saved_config(args, locals()))

    num_epoch_step = args.num_epoch_step
    num_episode = args.num_episode
    num_train_pi = args.num_train_pi
    num_train_v = args.num_train_v

    env = BitSelectEnv(num_env=args.num_env, num_bit=args.num_bit, num_subbit=args.num_subbit, file_dir=args.book_prefix, device=args.device, reward_step=args.reward_step, use_test=args.use_test, subref_size=args.subref_size,
                       subval_size=args.subval_size, subtest_size=args.subtest_size,
                       trainset_size=args.trainset_size, testset_size=args.testset_size)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    buffer = Buffer(obs_dim, act_dim, args.num_epoch_step, args.num_env, device=args.device)
    agent = Agent(obs_dim, act_dim, args.lr, args.device, args.discount_factor, target_kl=args.target_kl, ent_coef=args.ent_coef, rho=args.rho, clip_v=args.clip_v)
    logger.setup_pytorch_saver(agent)

    # candidate
    if args.use_candi:
        candi = Candidate(num_candi=args.num_candi, num_val_times=args.num_val_times)

    epoch = 0
    episode = 0
    interact_step = 0
    update_step = 0
    logger.log('-' * 20 + 'Start training' + '-' * 80, 'green')
    start_time = time.time()
    time_act = 0
    time_step = 0
    time_add = 0
    time_candi = 0
    time_update = 0
    time_all = 1e-7

    obs0 = env.reset()
    while True:
        for t in range(num_epoch_step):
            time_init = time.time()
            time_act_temp = 0
            time_step_temp = 0
            time_add_temp = 0
            time_candi_temp = 0
            time_update_temp = 0

            time_temp = time.time()
            action, logp, v = agent.get_action(obs0)
            time_act_temp += time.time() - time_temp

            time_temp = time.time()
            obs1, rewards, done, info = env.step(action)
            time_step_temp += time.time() - time_temp
            interact_step += 1

            time_temp = time.time()
            if 'last' in info.keys():
                episode += info['last'].get('episode', 0)
            if args.use_candi:
                if 'last' in info.keys():
                    candi.update(info['last']['state'], info['last']['map_train'], info['last']['map_test'], env.get_maps)
                    logger.store(MAP_LAST_TRAIN=candi.get_map_train())
                    logger.store(MAP_LAST_TEST=candi.get_map_test())
            time_candi_temp += time.time() - time_temp

            time_temp = time.time()
            buffer.add(obs0, action, rewards, done, v, logp)
            time_add_temp += time.time() - time_temp

            obs0 = obs1

            if t == num_epoch_step - 1:
                logger.store(**info['single'])
                for k, v in info['multi'].items():
                    for v_i in v:
                        logger.store(**{k: v_i})
                value = agent.compute_value(obs0)
                buffer.finish_buffer(value)
                batch = buffer.get()
                time_temp = time.time()
                update_infos = agent.compute_loss(batch, num_train_pi, num_train_v)
                time_update_temp += time.time() - time_temp

                update_step += 1
                logger.store(**update_infos)

            time_act += time_act_temp
            time_step += time_step_temp
            time_add += time_add_temp
            time_candi += time_candi_temp
            time_update += time_update_temp
            time_all += time.time() - time_init

        logger.log_tabular('Name', '{}-{}-{}'.format(args.dataset, args.hmethod, args.exp_name))
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('Episode', episode)
        logger.log_tabular('TotalEnvInteracts', interact_step)
        logger.log_tabular('TotalUpdateSteps', update_step)
        logger.log_tabular('Random', average_only=True)
        logger.log_tabular('NDomSet', average_only=True)
        logger.log_tabular('RL', average_only=True)
        logger.log_tabular('MAP_TRAIN', average_only=True)
        logger.log_tabular('MAP_TEST', average_only=True)
        if args.use_candi:
            logger.log_tabular('MAP_LAST_TRAIN', average_only=True)
            logger.log_tabular('MAP_LAST_TEST', average_only=True)
        logger.log_tabular('EpRet', with_min_and_max=True)
        # logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('V', average_only=True)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)

        logger.log_tabular('Time', time.time() - start_time)

        logger.log_tabular('time_act', time_act)
        logger.log_tabular('time_step', time_step)
        logger.log_tabular('time_add', time_add)
        logger.log_tabular('time_candi', time_candi)
        logger.log_tabular('time_update', time_update)
        logger.log_tabular('time_sum', time_all)

        logger.log_tabular('time_act_ratio', time_act / time_all)
        logger.log_tabular('time_step_ratio', time_step / time_all)
        logger.log_tabular('time_add_ratio', time_add / time_all)
        logger.log_tabular('time_candi_ratio', time_candi / time_all)
        logger.log_tabular('time_update_ratio', time_update / time_all)
        logger.log_tabular('time_sum_ratio', (time_act + time_step + time_add + time_candi + time_update) / time_all)

        logger.dump_tabular()
        data = {'agent': agent.get_param()}
        if args.use_candi:
            data['candi'] = candi.get_state_dict()
        logger.save_state(data)

        time_act = 0
        time_step = 0
        time_add = 0
        time_candi = 0
        time_update = 0
        time_all = 1e-7

        epoch += 1
        if episode > num_episode:
            break
