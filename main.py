import numpy as np
import torch
import argparse
import os
from scipy.io import loadmat, savemat
import utils
from Env import caosuanEnv
import COPO
from torch.utils.tensorboard import SummaryWriter
from eval_policy import eval_policy
writer = SummaryWriter("runs/")




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default="COPO")
    parser.add_argument("--env", default="caosuan")
    parser.add_argument("--seed", default=20, type=int)
    parser.add_argument("--eval_freq", default=50, type=int)
    parser.add_argument("--max_timesteps", default=2e6, type=int)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--load_model", default="")
    # TD3
    parser.add_argument("--expl_noise", default=0)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--discount", default=0.99)
    parser.add_argument("--tau", default=0.005)
    parser.add_argument("--policy_noise", default=0.2)
    parser.add_argument("--noise_clip", default=0.5)
    parser.add_argument("--policy_freq", default=2, type=int)

    parser.add_argument("--alpha", default=2.5)
    parser.add_argument("--normalize", default=True)
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    # load data
    actions =
    states =
    rewards =
    next_states =
    not_dones =
    costs =


    dataset = {'observations': np.array(states),
               'actions': np.array(actions),
               'next_observations': np.array(next_states),
               'rewards': np.array(rewards),
               'terminals': np.array(not_dones),
               'cost': np.array(costs)
               }

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = caosuanEnv()

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = 
    action_dim = 
    max_action = 

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq,
        "alpha": args.alpha
    }

    policy = COPO.COPO(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    replay_buffer.convert_caosuan(dataset)
    if args.normalize:
        mean, std = replay_buffer.normalize_states()
    else:
        mean, std = 0, 1
    evaluations = []
    episode_reward = []
    episode_cost = []
    episode_lagrange_multiplier = []
    for t in range(int(args.max_timesteps)):
        qloss = policy.train(replay_buffer, args.batch_size)
        eval_policy_reward, eval_policy_cost = eval_policy(policy, args.seed, mean, std)
        lagrange_multiplier = policy._lagrange.lagrangian_multiplier
        episode_reward.append(eval_policy_reward)
        episode_cost.append(eval_policy_cost)
        episode_lagrange_multiplier.append(lagrange_multiplier)
        file_name = 'Offline_COPO_PID_lagrangian_eval_policy_reward.mat'
        savemat(file_name,
                {'Offline_COPO_PID_lagrangian_eval_policy_reward': np.array(episode_reward).reshape(-1, 1)})
        file_name = 'Offline_COPO_PID_lagrange_multiplier.mat'
        savemat(file_name,
                {'Offline_COPO_PID_lagrangian_multiplier': np.array(episode_lagrange_multiplier).reshape(-1, 1)})

        writer.add_scalar('COPO_PID_Lagrangian_eval_policy/reward', eval_policy_reward, global_step=t)
        writer.add_scalar('COPO_PID_Lagrangian_eval_policy/cost', eval_policy_cost, global_step=t)
        writer.add_scalar('COPO_PID_lagrange_multiplier', lagrange_multiplier, global_step=t)
        if t % 10 == 0:
            writer.add_scalar('loss/Qloss', qloss, global_step=t)

        if (t + 1) % args.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            evaluations.append(eval_policy(policy, args.seed, mean, std))
            np.save(f"./results/{file_name}", evaluations)
        if args.save_model: policy.save(f"./models/{file_name}")
