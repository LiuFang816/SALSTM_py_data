from torch_rl.policies.Policy import Policy
from torch_rl.policies.DiscreteRandomPolicy import DiscreteRandomPolicy
import torch_rl
import gym
import numpy as np
from torch.autograd import Variable
import random

class DiscreteEpsilonGreedyPolicy(Policy):
    '''Chooses random action with probability epsilon'''

    def __init__(self, action_space, epsilon,policy):
        Policy.__init__(self, action_space)
        self.alpha=epsilon
        self.policy_other=policy
        self.policy_alpha=DiscreteRandomPolicy(action_space)


    # Must be called at the begining of a new episode
    def start_episode(self, **parameters):
        self.policy_alpha.start_episode(**parameters)
        self.policy_other.start_episode(**parameters)
        pass

    # Must be called at the end of a new episode, with an optionnal feedback over the whole episode
    def end_episode(self):
        self.policy_alpha.end_episode()
        self.policy_other.end_episode()


    # We assume that the observation is a 1xsingle observation
    def observe(self, observation):
        self.policy_alpha.observe(observation)
        self.policy_other.observe(observation)
        pass

    # Sample an action
    def sample(self):
        action_alpha=self.policy_alpha.sample()
        action_other=self.policy_other.sample()
        if (random.random()<self.alpha):
            self.alpha_chosen=True
            return action_alpha
        else:
            self.alpha_chosen=False
            return action_other
