#!/usr/bin/env python

"""
Almost identical to run_expert.py of Jonathon Ho
Just modified so that it can be used with pytorch instead of tensorflow

Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import numpy as np
import gym
import torch
from torch.autograd import Variable
from models import FCNet

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--save_path', type=str, default=None,
                        help='The path to save the expert rollouts to')
    args = parser.parse_args()

    print('loading and building expert policy')
    checkpoint = torch.load(args.expert_policy_file)
    model = checkpoint['factory'](checkpoint['factory_args'])
    model.load_state_dict(checkpoint['state_dict'])
    model.train(False)
    print('loaded and built')

    import gym
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    for i in range(args.num_rollouts):
        print('iter', i)
        iter_obs = []
        iter_actions = []
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = model(Variable(torch.from_numpy(obs[None,:]).float()))
            action = action.data.numpy()
            iter_obs.append(obs)
            iter_actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if True:
                env.render()
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)
        observations.append(iter_obs)
        actions.append(iter_actions)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    expert_data = {'observations': np.array(observations),
                   'actions': np.array(actions)}
    if args.save_path is not None:
        pickle.dump(expert_data, open(args.save_path, 'wb'))

if __name__ == '__main__':
    main()
