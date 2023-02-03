import gym
import numpy as np

import MPC_model as mpc
from sklearn.model_selection import train_test_split


def get_MPC_data(env: gym.Env, num_paths: int, horizon: int):
    paths = []
    for _ in range(num_paths):
        path_obs, path_next_obs, path_action, path_rewards, path_costs = [], [], [], [], []

        ep_step = 0
        obs = env.reset()
        while True:
            path_obs.append(obs)
            ep_step += 1
            action = env.action_space.sample()
            path_action.append(action)
            obs, reward, done, _ = env.step(action)
            path_next_obs.append(obs)
            path_rewards.append(reward)
            if done or ep_step >= horizon:
                break
        path = {
            "state": np.array(path_obs),
            "next_state": np.array(path_next_obs),
            "reward": np.array(path_rewards),
            "action": np.array(path_action),
        }
        paths.append(path)
    return paths


def compute_normalization(data):
    return data.mean(axis=0), data.std(axis=0)


# %%
env = gym.make("CartPole-v1")
paths = get_MPC_data(env=env, num_paths=50000, horizon=300)

all_s = np.concatenate([d["state"] for d in paths])
all_next_s = np.concatenate([d["next_state"] for d in paths])
all_delta_s = all_next_s - all_s
all_a = np.concatenate([d["action"] for d in paths])
all_r = np.concatenate([d["reward"] for d in paths])

mean_s, std_s = compute_normalization(all_s)
mean_delta_s, std_delta_s = compute_normalization(all_delta_s)
mean_a, std_a = compute_normalization(all_a)
mean_r, std_r = compute_normalization(all_r)
N = all_s.shape[0]

# normalize
norm_s = (all_s - mean_s) / (std_s + 1e-7)
norm_a = (all_a - mean_a) / (std_a + 1e-7)

## input: norm_s_a
# output: norm_delta_s
norm_s_a = np.column_stack((norm_s, norm_a))
norm_delta_s = ((all_next_s - all_s) - mean_delta_s) / (std_delta_s + 1e-7)

x_train, x_test, y_train, y_test = train_test_split(
    norm_s_a, norm_delta_s, test_size=0.2)

# train input/output data
args = mpc.Args()
args.epochs = 20
args.batch_size = 32
args.lr = 0.0001
args.in_dim = x_train.shape[1]
args.out_dim = y_train.shape[1]
args.n_hidden_1 = 128
args.n_hidden_2 = 64
args.data_x_train = x_train
args.data_y_train = y_train
args.data_x_val = x_test
args.data_y_val = y_test

mpc.train(args=args)
