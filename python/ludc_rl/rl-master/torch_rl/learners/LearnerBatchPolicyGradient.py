from torch_rl.learners import Learner, LearnerLog
from torch.autograd import Variable
from torch_rl.tools import Memory
import torch
from torch_rl.policies import DiscreteModelPolicy
import numpy as np

class LearnerBatchPolicyGradient(Learner):
    '''
    Allows to learn a policy by using the Policy Gradient technique with variance term (like LearnerPolicyGradient, but with batches)

    Params:
        - log: the log class for monitoring the learning algorithm
        - action_space(gym.spaces.Discrete): the action space of the resulting policy
        - average_reward_window: computes the average reward over the n last episodes and use this value as a vraiance term in REINFORCE
        - torch_model: the pytorch model taking as input an observation, and returning a probability for each possible action
        - optimizer: the SGD optimizer (using the torch_model parameters)
    '''
    def __init__(self,log=LearnerLog(),action_space=None,average_reward_window=10,torch_model=None,optimizer=None):
        self.torch_model=torch_model
        self.optimizer=optimizer
        self.average_reward_window=average_reward_window
        self.action_space=action_space
        self.log=log
        self.observation_shape=None

    def reset(self,**parameters):
        #Initilize the memories where the rewards (for each time step) will be stored
        self.memory_past_rewards=[]
        pass

    def sample_action(self,obs):
        probabilities = self.torch_model(Variable(obs,requires_grad=True))
        a = probabilities.multinomial(1)
        return (a)



    def create_observation(self,nb):
        ns=(nb,)+self.observation_shape
        return torch.zeros(ns)

    def step(self,envs=None,discount_factor=1.0,maximum_episode_length=100,render=False):
        '''
        Computes the gradient descent over one set of enviroments

        Params:
            - envs: a list of openAI environments (same sensor, same actions set) that will be sampled in parallel
            - discount_factor: the discount factor used for REINFORCE
            - maximum_episode_length: the maximum episode length before stopping the episode
        '''
        nb = len(envs)

        self.actions_taken=[]
        self.rewards = []
        self.finished=[]
        for i in range(nb):
            self.finished.append(-1)

        #self.observation is a batch matrix
        current_observation=None
        for i in range(nb):
            observation=envs[i].reset()

            if ((i==0) and render):
                envs[i].render()


            if (self.observation_shape is None):
                if (isinstance(observation,torch.Tensor)):
                    observation=observation.numpy()
                self.observation_shape=observation.shape

            if (i==0):
                current_observation=self.create_observation(nb)

            if (isinstance(observation, np.ndarray)):
                observation = torch.Tensor(observation)

            current_observation[i].copy_(observation)


        #Draw the episode
        sum_rewards=np.zeros(nb)

        for t in range(maximum_episode_length):
            self.rewards.append([])
            actions = self.sample_action(current_observation)
            self.actions_taken.append(actions)
            last_observation=current_observation

            current_observation=current_observation.clone()

            for i in range(nb):
                if (self.finished[i]>=0):
                    self.rewards[t].append(0)
                    pass
                else:
                    observation,immediate_reward,fin,info=envs[i].step(actions.data[i][0])

                    if ((i == 0) and render):
                        envs[i].render()

                    if (isinstance(observation, np.ndarray)):
                        observation = torch.Tensor(observation)
                    sum_rewards[i]=sum_rewards[i]+immediate_reward
                    self.rewards[t].append(immediate_reward)
                    current_observation[i].copy_(observation)
                    if (fin):
                        self.finished[i]=t+1
            #If all the environments are finished
            if (np.equal(self.finished,-1).sum()==0):
                break

        for i in range(nb):
            if (self.finished[i]==-1):
                self.finished[i]=maximum_episode_length

        self.log.new_iteration()
        self.log.add_dynamic_value("avg_total_reward", sum_rewards.mean())
        self.log.add_dynamic_value("avg_length", np.mean(self.finished))

        #Update the policy
        T=len(self.actions_taken)

        Tm = len(self.memory_past_rewards)
        for t in range(Tm, T):
            self.memory_past_rewards.append(Memory(self.average_reward_window))

        # Update memory with (future) discounted rewards
        discounted_reward = np.zeros(nb)
        self.optimizer.zero_grad()
        grads = []
        for t in range(T, 0, -1):
            discounted_reward = discounted_reward * discount_factor + self.rewards[t - 1]
            for i in range(nb):
                if (self.finished[i]>=t):
                    self.memory_past_rewards[t - 1].push(discounted_reward[i])

            avg_reward = self.memory_past_rewards[t - 1].mean()
            rein=torch.Tensor(discounted_reward)
            for i in range(nb):
                if (t>self.finished[i]):
                    assert rein[i]==0

            self.actions_taken[t - 1].reinforce(rein)
            grads.append(None)
        torch.autograd.backward(self.actions_taken, grads)
        self.optimizer.step()

    def get_policy(self,stochastic=False):
        '''
        Params:
            - stochastic: True if one wants a stochastic policy, False if one wants a policy based on the argmax of the output probabilities
        '''
        assert stochastic==False
        return DiscreteModelPolicy(self.action_space,self.torch_model)