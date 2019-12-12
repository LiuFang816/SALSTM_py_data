import torch.nn as nn
import torch.optim as optim
import torch_rl.learners as learners
import torch
import gym
from torch_rl.tools import rl_evaluate_policy, rl_evaluate_policy_multiple_times


LEARNING_RATE=0.001
STDV=0.01
LATENT_SIZE=10

env=gym.make('CartPole-v0')

#Creation of the policy
A = env.action_space.n
N = env.observation_space.high.size
print("Number of Actions is: %d" % A)

#===================================================================== CREATION OF THE POLICY =============================================
class ActionModel(nn.Module):
    def __init__(self, state_size,action_size):
        super(ActionModel, self).__init__()
        self.linear=nn.Linear(state_size, action_size)
        self.softmax=nn.Softmax()
        self.linear.weight.data.normal_(0,STDV)
        self.linear.bias.data.normal_(0,STDV)

    def forward(self, data):
        output = self.linear(data)
        output=self.softmax(output)
        return output

class RecurrentModel(nn.Module):
    def __init__(self, state_size,observation_size,action_size):
        super(RecurrentModel, self).__init__()
        self.linear_state=nn.Linear(state_size, state_size)
        self.linear_observation=nn.Linear(observation_size, state_size)
        self.linear_action=nn.Linear(action_size, state_size)
        self.softmax=nn.Softmax()
        self.linear_state.weight.data.normal_(0,STDV)
        self.linear_state.bias.data.normal_(0,STDV)
        self.linear_observation.weight.data.normal_(0,STDV)
        self.linear_observation.bias.data.normal_(0,STDV)
        self.linear_action.weight.data.normal_(0,STDV)
        self.linear_action.bias.data.normal_(0,STDV)
        self.t1=nn.Tanh()
        self.t2=nn.Tanh()
        self.t3=nn.Tanh()
        self.t4=nn.Tanh()


    def forward(self, state,observation,action):
        s=self.t1(self.linear_state(state))
        o=self.t2(self.linear_observation(observation))
        a=self.t3(self.linear_action(action))
        output=self.t4(s+o+a)
        return output

class AllModel(nn.Module):
    def __init__(self, m1,m2):
        super(AllModel, self).__init__()
        self.m1=m1
        self.m2=m2

model_action = ActionModel(LATENT_SIZE,A)
model_recurrent = RecurrentModel(LATENT_SIZE,N,A)
am=AllModel(model_action,model_recurrent)
optimizer= optim.Adam(am.parameters(), lr=LEARNING_RATE)


learning_algorithm=learners.LearnerRecurrentPolicyGradient(action_space=env.action_space,average_reward_window=10,torch_model_action=model_action,torch_model_recurrent=model_recurrent,initial_state=torch.zeros(1,LATENT_SIZE),optimizer=optimizer)
learning_algorithm.reset()
while(True):
    learning_algorithm.step(env=env,discount_factor=0.9,maximum_episode_length=100)
    policy=learning_algorithm.get_policy()
    r=rl_evaluate_policy_multiple_times(env,policy,100,1.0,10)
    print("Evaluation avg reward = %f "% r)

    print("Reward = %f"%learning_algorithm.log.get_last_dynamic_value("total_reward"))







