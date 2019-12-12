from torch_rl.policies.Policy import Policy
import torch_rl
import gym
import numpy as np
from torch.autograd import Variable
import torch
class DiscreteModelPolicy(Policy):
    '''
    A DiscreteModelPolicy is a policy that computes one score for each possible action given a particular torch model, and returns the action with the max value

    Params:
     - action_space: the action space
     - torch_model: the observation->action model
     - stochastic: if True, it assumes that the output of the torch_model is a distribution, and samples over this distribution
    '''

    def __init__(self, action_space, torch_model,stochastic=False):
        Policy.__init__(self, action_space)
        assert isinstance(action_space, gym.spaces.Discrete), "In DiscreteModelPolicy, the action space must be discrete"
        #assert isinstance(sensor_space, torch_rl.spaces.PytorchBox), "In DeeQPolicy, sensor_space must be a PytorchBox"

        self.action_space = action_space
        #self.sensor_space = sensor_space
        self.torch_model=torch_model
        self.stochastic=stochastic

    # We assume that the observation is a 1xsingle observation
    def observe(self, observation):
        if (isinstance(observation, np.ndarray)):
            observation = torch.Tensor(observation)
        self.observation=observation.unsqueeze(0)
        pass

    # Sample an action
    def sample(self):
        scores=self.torch_model(Variable(self.observation))
        if (self.stochastic):
            scores=scores.multinomial(1)
            action=scores.data[0][0]
            return action

        action=scores.max(1)
        action=action[1].data
        action=action[0][0]
        return action
