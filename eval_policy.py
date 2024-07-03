import numpy as np
from Env import caosuanEnv


def eval_policy(policy, seed, mean, std, seed_offset=100, eval_episodes=1):
    eval_env = caosuanEnv()
    eval_env.seed(seed + seed_offset)
    avg_reward = 0.
    avg_cost = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            state = (np.array(state).reshape(1, -1) - mean) / std
            action = policy.select_action(state)
            ary_action = action # need to scale the original [0,1] action to original action
            state, reward, done, cost, _ = eval_env.step(ary_action)
            avg_reward += reward
            avg_cost += cost
            print("reward:", reward)
            print("action:", ary_action)

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {reward:.3f}")
    print("---------------------------------------")
    return reward, avg_cost
