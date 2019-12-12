from torch_rl.policies.Policy import Policy
import torch_rl
import gym
import numpy as np
from torch.autograd import Variable

class AlphaMixturePolicy(Policy):
    '''Chooses policy_alpha with probability alpha, and policy_other with probability 1-alpha'''

    def __init__(self, action_space, alpha,policy_alpha,policy_other):
        Policy.__init__(self, action_space)
        self.alpha=alpha
        self.policy_alpha=policy_alpha
        self.policy_other=policy_other


    # Must be called at the begining of a new episode
    def start_episode(self, **parameters):
        self.policy_alpha.start_episode(**parameters)
        self.policy_other.start_episode(**parameters)
        pass

    # Must be called at the end of a new episode, with an optionnal feedback over the whole episode
    def end_episode(self):
        #WARNING: the feedback is provided to the two policies while policy_alpha has been used only for some actions. Can cause a big problem during learning
        self.policy_alpha.end_episode()
        self.policy_other.end_episode()

    # We assume that the observation is a 1xsingle observation
    def observe(self, observation):
        self.policy_alpha.observe(observation)
        self.policy_other.observe(observation)

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
