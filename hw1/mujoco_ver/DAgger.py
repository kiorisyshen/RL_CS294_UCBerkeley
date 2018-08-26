import pickle
import tensorflow as tf
import numpy as np
import random
import tf_util
import gym
import load_policy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('expert_policy_file', type=str)
parser.add_argument('envname', type=str)
parser.add_argument('--render', action='store_true')
parser.add_argument("--max_timesteps", type=int)
parser.add_argument('--num_rollouts', type=int, default=250,
                    help='Number of expert roll outs')
args = parser.parse_args()


max_epoch = 5

class BC_net(nn.Module):
    def __init__(self, inputdim, outputdim, b_size):
        super(BC_net, self).__init__()
        self.indim = inputdim
        self.outdim = outputdim
        self.fc1 = nn.Linear(inputdim, inputdim*2)
        self.fc2 = nn.Linear(inputdim*2, outputdim*2)
        self.fc3 = nn.Linear(outputdim*2, outputdim)
        self.batch_size = b_size

    def forward(self, x):
        x = x.view(self.batch_size, self.indim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

def main():
    # train our policy using Data Aggressive
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')
    
    with tf.Session():
        tf_util.initialize()
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit
        
        net = BC_net(len(env.observation_space.high), len(np.squeeze(env.action_space.high)), 1)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(net.parameters(), lr=0.01)
        loss_list = []

        plt.ion()
        steps_all = 0
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            done_counter = 0
            steps = 0

            while not done and done_counter < 200:
                net.zero_grad()
                output = net(torch.tensor(obs[None,:], dtype=torch.float32))
                lbl = policy_fn(obs[None,:])
                loss = criterion(output, torch.tensor(lbl, dtype=torch.float32))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.data.numpy())

                obs, r, done, _ = env.step(np.squeeze(output.data.numpy()))
                steps += 1
                steps_all += 1
                if done:
                    done_counter += 1

                if args.render and i > 120:
                    env.render()

                if steps+1 % 1000 == 0:
                    break
            plt.clf()
            plt.plot(range(steps_all), loss_list, 'b-')
            plt.ylim(0, 10)
            plt.pause(.05)
            plt.draw()

if __name__ == '__main__':
    main()