import pickle
import tensorflow as tf
import numpy as np
import random
import tf_util
import gym
import load_policy
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('expert_policy_file', type=str)
parser.add_argument('envname', type=str)
parser.add_argument('--render', action='store_true')
parser.add_argument("--max_timesteps", type=int)
parser.add_argument('--num_rollouts', type=int, default=3,
                    help='Number of expert roll outs')
args = parser.parse_args()

def iter_batch(data, lbl, batchsize=4, rand=True):
    if rand:
        c = list(zip(data, lbl))
        random.shuffle(c)

    d, l = zip(*c)

    for i in range(int(len(data)/batchsize)):
        yield d[i*batchsize:(i+1)*batchsize], l[i*batchsize:(i+1)*batchsize]

batchsize = 4
max_epoch = 5

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as d

class BC_net(nn.Module):
    def __init__(self, inputdim, outputdim):
        super(BC_net, self).__init__()
        self.indim = inputdim
        self.outdim = outputdim
        self.fc1 = nn.Linear(inputdim, inputdim*2)
        self.fc2 = nn.Linear(inputdim*2, outputdim*2)
        self.fc3 = nn.Linear(outputdim*2, outputdim)

    def forward(self, x):
        x = x.view(-1, self.indim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

def main():
    # train our policy using Behavior clone
    expert_data = pickle.load( open("experts_traindata/"+args.envname, "rb") )
    observs = np.squeeze(expert_data['observations']).astype(np.float32)
    actions = np.squeeze(expert_data['actions']).astype(np.float32)

    net = BC_net(len(observs[0]), len(np.squeeze(actions[0])))
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    iter_count = 0
    loss_list = []

    plt.ion()
    for epoch in range(max_epoch):
        iter_ = iter_batch(observs, actions)
        for data, lbl in iter_:
            net.zero_grad()
            output = net(torch.tensor(data, dtype=torch.float32))
            # output = torch.clamp(output, -1, 1)
            loss = criterion(output, torch.tensor(lbl, dtype=torch.float32))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_list.append(loss.data.numpy())
            iter_count += 1

            # if iter_count%1000 == 0:
            #     plt.clf()
            #     plt.plot(range(iter_count), loss_list, 'b-')
            #     plt.pause(.05)
            #     plt.draw()
        plt.clf()
        plt.plot(range(iter_count), loss_list, 'b-')
        plt.pause(.05)
        plt.draw()
        print(epoch, iter_count)

    # see how our learned policy works
    with tf.Session():
        tf_util.initialize()
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []

        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                # action = policy_fn(obs[None,:])
                output = net(torch.tensor(obs, dtype=torch.float32))

                obs, r, done, _ = env.step(np.squeeze(output.data.numpy()))
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))


if __name__ == '__main__':
    main()