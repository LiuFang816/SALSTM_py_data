from .Learner import Learner
from torch.autograd import Variable
from torch_rl.tools import Memory
import torch
from torch_rl.policies import DiscreteModelPolicy
from .LearnerLog import *
import logging
import torch.nn as nn
import random
import numpy as np

class LearnerDQN(Learner):
    '''
    Allows to learn a policy by using the Policy Gradient technique with variance term

    Params:
        - log: the log class for monitoring the learning algorithm
        - action_space (gym.spaces.Discrete): the action space of the resulting policy
        - sensor_space (PytorchBox): the sensor space
        - size_batch_replay_memory: The size of each experience replay memory batch
        - nb_batch_replay_memory: the number of replay batches. nb_batch_replay_memory*size_batch_replay_memory = size of replay memory
        - torch_model: the underlying model that takes an observation and compute a score for each action
        - exploration_policy: the policy used for exaploration. Typically, it is a epsilon-greedy policy based on torch_model
        - optimizer: the SGD optimizer (using the torch_model parameters)
    '''
    def __init__(self,log=LearnerLog(),action_space=None,observation_space=None,size_batch_replay_memory=None,nb_batch_replay_memory=None,torch_model=None,exploration_policy=None,optimizer=None):
        assert (size_batch_replay_memory > 0)
        assert (nb_batch_replay_memory > 0)
        self.log=log
        self.action_space=action_space
        self.size_batch_replay_memory = size_batch_replay_memory
        self.nb_batch_replay_memory = nb_batch_replay_memory
        self.exploration_policy = exploration_policy
        self.torch_model = torch_model
        self.action_space = action_space
        self.observation_space = observation_space

        self.optimizer = optimizer
        self.criterion = nn.MSELoss()

        # Creation of the experience replay memory
        self.st0_batches = []
        self.st1_batches = []
        self.reward_batches = []
        self.action_batches = []
        self.finished_batches = []

        logging.info("Creating experience replay batches in memory....")
        # The batches will be randomly filled during execution of the policy
        shape = (size_batch_replay_memory,) + self.observation_space.low.shape
        for i in range(self.nb_batch_replay_memory):
            self.st0_batches.append(torch.Tensor(torch.Size(shape)))
            self.st1_batches.append(torch.Tensor(torch.Size(shape)))
            self.reward_batches.append(torch.Tensor(torch.Size([size_batch_replay_memory, 1])))
            self.action_batches.append(torch.ByteTensor(torch.Size([size_batch_replay_memory, self.action_space.n])))
            self.finished_batches.append(torch.Tensor(torch.Size([size_batch_replay_memory, 1])))

        # Right now, the batches are empty and must be filled before strating to learn
        self.batches_still_empty = True
        self.batches_fill_idx = 0
        self.batches_fill_in_idx = -1
        self.last_observation = None
        self.last_reward = None
        self.time = 0

    def write_index(self):
        if (self.batches_still_empty):
            self.batches_fill_in_idx = self.batches_fill_in_idx + 1
            if (self.batches_fill_in_idx >= self.size_batch_replay_memory):
                self.batches_fill_idx = self.batches_fill_idx + 1
                self.batches_fill_in_idx = -1
                if (self.batches_fill_idx >= self.nb_batch_replay_memory):
                    self.batches_still_empty = False
                return self.write_index()
            else:
                return (self.batches_fill_idx, self.batches_fill_in_idx)
        else:
            idx = random.randint(0, self.nb_batch_replay_memory - 1)
            iidx = random.randint(0, self.size_batch_replay_memory - 1)
            return (idx, iidx)

    def reset(self,**parameters):
        pass

    def updateOnBatch(self,discount_factor):
        self.optimizer.zero_grad()
        idx_batch = random.randint(0, self.nb_batch_replay_memory - 1)
        # print("Updating policy on batch %d "%idx_batch)

        # Computing the output of the model on the whole batch
        Qs = self.torch_model(Variable(self.st0_batches[idx_batch],requires_grad=True))
        Q = Qs.masked_select(Variable(self.action_batches[idx_batch],requires_grad=False))
        out = Variable(self.st1_batches[idx_batch])
        Qprime = self.torch_model(out)
        Qprime = Qprime.max(1)[0]
        Qprime = Variable(self.reward_batches[idx_batch]) + discount_factor * Variable(
            self.finished_batches[idx_batch]) * Qprime
        Qprime=Qprime.detach()
        loss = self.criterion(Q, Qprime)
        #print("Loss = %f"%loss.data[0])
        loss.backward()

        self.optimizer.step()

    # We assume that the observation is a 1xsingle observation
    def add_observation(self,last_observation, observation,reward,action):
        self.last_idx, self.last_iidx = self.write_index()
        # print("Writing in position %d,%d" % (self.last_idx,self.last_iidx))
        self.st0_batches[self.last_idx][self.last_iidx].copy_(last_observation)
        self.st1_batches[self.last_idx][self.last_iidx].copy_(observation)
        self.reward_batches[self.last_idx][self.last_iidx][0] = reward
        self.action_batches[self.last_idx][self.last_iidx].fill_(0)
        self.action_batches[self.last_idx][self.last_iidx][action] = 1
        self.finished_batches[self.last_idx][self.last_iidx][0] = 1

    def step(self,env=None,discount_factor=1.0,maximum_episode_length=100):
        '''
        Update the memory with one episode, then make gradient descent over one random batch. Note that the mdel is updated when all the batches have been filled (burn in period)

        Params:
            - env: the openai Environment
            - discount_factor: the discount factor used for DQN
            - maximum_episode_length: the maximum episode length before stopping the episode
        '''
        self.actions_taken=[]
        self.rewards = []


        self.observation = env.reset()
        if (isinstance(self.observation,np.ndarray)):
            self.observation=torch.Tensor(self.observation)
        self.exploration_policy.observe(self.observation)
        #Draw the episode
        sum_reward=0
        for t in range(maximum_episode_length):
            action = self.exploration_policy.sample()
            observation,immediate_reward,finished,info=env.step(action)
            if (isinstance(observation, np.ndarray)):
                observation = torch.Tensor(observation)
            sum_reward=sum_reward+immediate_reward
            self.rewards.append(immediate_reward)
            self.add_observation(self.observation,observation,immediate_reward,action)
            self.observation=observation
            self.exploration_policy.observe(observation)
            if (finished):
                break

        self.log.new_iteration()
        self.log.add_dynamic_value("total_reward",sum_reward)

        if (not self.batches_still_empty):
            self.updateOnBatch(discount_factor)


    def get_policy(self):
        return DiscreteModelPolicy(self.action_space,self.torch_model)