from torch_rl.learners import Learner,LearnerLog
from torch.autograd import Variable
from torch_rl.tools import Memory
import torch
import numpy as np
import gym.spaces
from torch_rl.policies import Policy

class LearnerRecurrentPolicyGradient(Learner):
    '''
    Allows to learn a policy by using the Policy Gradient technique with variance term.
    The policy is represented by a recurrent NN.

    Params:
        - log: the log class for monitoring the learning algorithm
        - action_space(gym.spaces.Discrete): the action space of the resulting policy
        - average_reward_window: computes the average reward over the n last episodes and use this value as a vraiance term in REINFORCE
        - torch_model_action: the pytorch model taking as input an observation, and returning a probability for each possible action
        - torch_model_recurrent: the pytorch recurrent model (state,observation,action)-> new state. action is one hot vector.
        - initial_state: the state h_0
        - optimizer: the SGD optimizer (using the torch_model parameters)
    '''
    def __init__(self,log=LearnerLog(),action_space=None,average_reward_window=10,torch_model_action=None,torch_model_recurrent=None,initial_state=None,optimizer=None,cuda=False):
        self.optimizer=optimizer
        self.average_reward_window=average_reward_window
        self.action_space=action_space
        self.cuda=cuda

        #Creation of the one hot vectors for actions inputs
        self.actions_vectors=[]
        for a in range(self.action_space.n):
            v=torch.zeros(1,self.action_space.n)
            v[0][a]=1
            if (self.cuda):
                v=v.cuda()
            self.actions_vectors.append(Variable(v))


        self.initial_state=initial_state

        self.torch_model_recurrent=torch_model_recurrent
        self.torch_model_action=torch_model_action

        self.log=log

    def reset(self,**parameters):
        #Initilize the memories where the rewards (for each time step) will be stored
        self.memory_past_rewards=[]
        pass

    def sample_action(self):
        probabilities = self.torch_model_action(self.state)
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
        self.optimizer.zero_grad()



        self.state=Variable(self.initial_state)

        self.actions_taken=[]
        self.rewards = []

        self.observation=env.reset()
        if (render):
            env.render()
        if (isinstance(self.observation,np.ndarray)):
            self.observation=torch.Tensor(self.observation)

        if (self.cuda):
            self.observation=self.observation.cuda()

        self.observation=Variable(self.observation.unsqueeze(0))
        va=torch.zeros(1,self.action_space.n)
        if (self.cuda):
            va=va.cuda()
        self.state=self.torch_model_recurrent(self.state,self.observation,Variable(va))
        sum_reward=0
        for t in range(maximum_episode_length):
            action = self.sample_action()
            self.observation,immediate_reward,finished,info=env.step(action)
            if (render):
                env.render()

            if (isinstance(self.observation, np.ndarray)):
                self.observation = torch.Tensor(self.observation)
            if (self.cuda):
                self.observation=self.observation.cuda()

            self.observation = Variable(self.observation.unsqueeze(0))
            sum_reward=sum_reward+immediate_reward
            self.rewards.append(immediate_reward)
            if (finished):
                break
            self.state = self.torch_model_recurrent(self.state, self.observation,self.actions_vectors[action])

        self.log.new_iteration()
        self.log.add_dynamic_value("total_reward",sum_reward)

        #Update the policy
        T = len(self.actions_taken)
        Tm = len(self.memory_past_rewards)
        for t in range(Tm, T):
            self.memory_past_rewards.append(Memory(self.average_reward_window))

        # Update memory with (future) discounted rewards
        discounted_reward = 0
        grads = []
        for t in range(T, 0, -1):
            discounted_reward = discounted_reward * discount_factor + self.rewards[t - 1]
            self.memory_past_rewards[t - 1].push(discounted_reward)
            avg_reward = self.memory_past_rewards[t - 1].mean()
            rein=torch.Tensor([[discounted_reward - avg_reward]])
            if (self.cuda):
                rein=rein.cuda()

            self.actions_taken[t - 1].reinforce(rein)
            grads.append(None)
        torch.autograd.backward(self.actions_taken, grads)
        #for p in self.torch_model_recurrent.parameters():
        #    print(p.grad.data)
        self.optimizer.step()


    def get_policy(self,stochastic=False):
        '''
        Params:
            - stochastic: True if one wants a stochastic policy, False if one wants a policy based on the argmax of the output probabilities
        '''
        return DiscreteRecurrentModelPolicy(self.action_space,self.torch_model_action,self.torch_model_recurrent,self.initial_state,stochastic=stochastic,cuda=self.cuda)


class DiscreteRecurrentModelPolicy(Policy):
    '''A Discrete policy based on a recurrent neural network'''

    def __init__(self, action_space, torch_model_action,torch_model_recurrent,initial_state,stochastic=False,cuda=False):
        Policy.__init__(self, action_space)
        assert isinstance(action_space, gym.spaces.Discrete), "In DiscreteModelPolicy, the action space must be discrete"
        #assert isinstance(sensor_space, torch_rl.spaces.PytorchBox), "In DeeQPolicy, sensor_space must be a PytorchBox"

        self.action_space = action_space
        #self.sensor_space = sensor_space
        self.torch_model_action=torch_model_action
        self.torch_model_recurrent=torch_model_recurrent
        self.initial_state=initial_state
        self.stochastic=stochastic
        self.cuda=cuda

        self.actions_vectors = []
        for a in range(self.action_space.n):
            v = torch.zeros(1, self.action_space.n)
            if (self.cuda):
                v=v.cuda()
            v[0][a] = 1
            self.actions_vectors.append(Variable(v))

    def observe(self, observation):
        if (isinstance(observation, np.ndarray)):
            observation = torch.Tensor(observation)
        if (self.cuda):
            observation=observation.cuda()

        o=Variable(observation.unsqueeze(0))
        self.state=self.torch_model_recurrent(self.state,o,self.last_action)
        pass

    def sample(self):
        scores=self.torch_model_action(self.state)
        if (self.stochastic):
            scores=scores.multinomial(1)
            action=scores.data[0][0]
            self.last_action = self.actions_vectors[action]
            return action

        action=scores.max(1)
        action=action[1].data
        action=action[0][0]

        self.last_action=self.actions_vectors[action]
        return action


    def start_episode(self, **parameters):
        self.state=Variable(self.initial_state)
        self.last_action=Variable(torch.zeros(1,self.action_space.n))
        if (self.cuda):
            self.last_action=self.last_action.cuda()
