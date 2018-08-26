#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os
import inspect
import random

import gym
import numpy as np
import pybullet_envs
import time
import pickle
import matplotlib.pyplot as plt

from torch import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def iter_batch(data, lbl, batchsize=4, rand=True):
    if rand:
        c = list(zip(data, lbl))
        random.shuffle(c)

    d, l = zip(*c)

    for i in range(int(len(data)/batchsize)):
        yield d[i*batchsize:(i+1)*batchsize], l[i*batchsize:(i+1)*batchsize]

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

num_rollouts = 20
max_steps = 500
envrender = True
batchsize = 4
max_epoch = 7
use_gpu = True
dtype_ = torch.FloatTensor


def main():
    # train our policy using Behavior clone
    expert_data = pickle.load( open("train_data/HopperBulletEnv-v0.pkl", "rb") )
    observs = np.squeeze(expert_data['observations']).astype(np.float32)
    actions = np.squeeze(expert_data['actions']).astype(np.float32)

    net = BC_net(len(observs[0]), len(np.squeeze(actions[0])))
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    if use_gpu:
        torch.backends.cudnn.benchmark=True
        dtype_ = torch.cuda.FloatTensor
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        net = net.cuda()
        criterion = criterion.cuda()

    iter_count = 0
    loss_list = []
    plt.ion()
    for epoch in range(max_epoch):
        iter_ = iter_batch(observs, actions)
        for data, lbl in iter_:
            net.zero_grad()
            output = net(torch.tensor(data, dtype=torch.float32).type(dtype_))
            # output = torch.clamp(output, -1, 1)
            loss = criterion(output, torch.tensor(lbl, dtype=torch.float32).type(dtype_))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if use_gpu:
                loss = loss.data.cpu()
            else:
                loss = loss.data
            loss_list.append(loss.numpy())
            iter_count += 1

            # if iter_count%1000 == 0:
                # plt.clf()
                # plt.plot(range(iter_count), loss_list, 'b-')
                # plt.ylim(0, 10)
                # plt.pause(.02)
                # plt.draw()
        print(epoch, iter_count)

    
    env = gym.make("HopperBulletEnv-v0")
    if envrender:
        env.render()

    returns = []
    for i in range(num_rollouts):
        print('iter', i)
        #disable rendering during reset, makes loading much faster
        obs = env.reset()
    
        done = False
        totalr = 0.
        steps = 0

        while not done:
            output = net(torch.tensor(obs, dtype=torch.float32).type(dtype_))

            if use_gpu:
                output = output.data.cpu()
            else:
                output = output.data

            obs, r, done, _ = env.step(np.squeeze(output.numpy()))
            totalr += r
            steps += 1

            if envrender:
                time.sleep(1./60.)
                still_open = env.render()
                if still_open==False:
                    return

            if steps % 100 == 0: print("%i/%i reward:%.2f"%(steps, max_steps, totalr))
            if steps >= max_steps:
                break

        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

if __name__=="__main__":
    main()
