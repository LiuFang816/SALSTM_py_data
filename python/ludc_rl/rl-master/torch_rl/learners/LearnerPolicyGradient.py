from .Learner import Learner
from torch.autograd import Variable
from torch_rl.tools import Memory
import torch
from torch_rl.policies import DiscreteModelPolicy
from .LearnerLog import *
import numpy as np

class LearnerPolicyGradient(Learner):
    '''
    Allows to learn a policy by using the Policy Gradient technique with variance term

    Params:
        - log: the log class for monitoring the learning algorithm
        - action_space(gym.spaces.Discrete): the action space of the resulting policy
        - average_reward_window: computes the average reward over the n last episodes and use this value as a vraiance term in REINFORCE
        - torch_model: the pytorch model taking as input an observation, and returning a probability for each possible action
        - optimizer: the SGD optimizer (using the torch_model parameters)
    '''
    def __init__(self,log=LearnerLog(),action_space=None,average_reward_window=10,torch_model=None,optimizer=None,entropy_coefficient=0.0):
        self.torch_model=torch_model
        self.optimizer=optimizer
        self.average_reward_window=average_reward_window
        self.action_space=action_space
        self.log=log
        self.entropy_coefficient=entropy_coefficient


    def reset(self,**parameters):
        #Initilize the memories where the rewards (for each time step) will be stored
        self.memory_past_rewards=[]
        pass

    def sample_action(self):
        obs=self.observation.unsqueeze(0)
        probabilities = self.torch_model(Variable(obs,requires_grad=True))
        ent=probabilities*probabilities.log()
        ent=ent.sum()
        if (self.entropy is None):
            self.entropy=ent
        else:
            self.entropy=self.entropy+ent


        a = probabilities.multinomial(1)
        self.actions_taken.append(a)
        return (a.data[0][0])

    def step(self,env=None,discount_factor=1.0,maximum_episode_length=100,render=False):
        '''
        Computes the gradient descent over one episode

        Params:
            - env: the openAI environment
            - discount_factor: the discount factor used for REINFORCE
            - maximum_episode_length: the maximum episode length before stopping the episode
        '''
        self.actions_taken=[]
        self.rewards = []

        self.observation=env.reset()
        if (render):
            env.render()
        if (isinstance(self.observation,np.ndarray)):
            self.observation=torch.Tensor(self.observation)

        self.entropy=None
        #Draw the episode
        sum_reward=0
        for t in range(maximum_episode_length):
            action = self.sample_action()
            self.observation,immediate_reward,finished,info=env.step(action)
            if (render):
                env.render()
            if (isinstance(self.observation, np.ndarray)):
                self.observation = torch.Tensor(self.observation)
            sum_reward=sum_reward+immediate_reward
            self.rewards.append(immediate_reward)
            if (finished):
                break

        self.log.new_iteration()
        self.log.add_dynamic_value("total_reward",sum_reward)

        #Update the policy
        T = len(self.actions_taken)
        Tm = len(self.memory_past_rewards)
        for t in range(Tm, T):
            self.memory_past_rewards.append(Memory(self.average_reward_window))

        # Update memory with (future) discounted rewards
        discounted_reward = 0
        self.optimizer.zero_grad()
        grads = []
        for t in range(T, 0, -1):
            discounted_reward = discounted_reward * discount_factor + self.rewards[t - 1]
            self.memory_past_rewards[t - 1].push(discounted_reward)
            avg_reward = self.memory_past_rewards[t - 1].mean()
            rein=torch.Tensor([[discounted_reward - avg_reward]])
            self.actions_taken[t - 1].reinforce(rein)
            grads.append(None)

        torch.autograd.backward(self.actions_taken, grads,retain_variables=True)
        self.entropy=self.entropy/T
        print("Entropy is %f" % self.entropy.data[0])

        if (self.entropy_coefficient>0):
            self.entropy=self.entropy*self.entropy_coefficient
            self.entropy.backward()
        self.optimizer.step()


    def get_policy(self,stochastic=False):
        '''
        Params:
            - stochastic: True if one wants a stochastic policy, False if one wants a policy based on the argmax of the output probabilities
        '''
        return DiscreteModelPolicy(self.action_space,self.torch_model,stochastic=stochastic)