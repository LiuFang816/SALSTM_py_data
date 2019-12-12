import torch.nn as nn
import torch.optim as optim
import torch_rl.learners as learners
import gym
from torch_rl.tools import rl_evaluate_policy, rl_evaluate_policy_multiple_times
from torch_rl.policies import DiscreteModelPolicy

LEARNING_RATE=0.01
STDV=0.01

env=gym.make('CartPole-v0')

#Creation of the policy
A = env.action_space.n
N = env.observation_space.high.size
print("Number of Actions is: %d" % A)

#===================================================================== CREATION OF THE POLICY =============================================
# Creation of a learning model Q(s): R^N -> R^A
class MyModel(nn.Module):
    def __init__(self, data_size, hidden_size,output_size):
        super(MyModel, self).__init__()
        self.linear=nn.Linear(data_size, hidden_size)
        self.tanh=nn.Tanh()
        self.linear2=nn.Linear(hidden_size,output_size)
        self.softmax=nn.Softmax()
        self.linear.weight.data.normal_(0,STDV)
        self.linear.bias.data.normal_(0,STDV)
        self.linear2.weight.data.normal_(0,STDV)
        self.linear2.bias.data.normal_(0,STDV)
        self.data_size=data_size

    def forward(self, data):
        output = self.linear(data)
        output=self.tanh(output)
        output=self.linear2(output)
        output=self.softmax(output)
        return output

model = MyModel(N,N*2,A)
optimizer= optim.Adam(model.parameters(), lr=LEARNING_RATE)

#policy=DiscreteModelPolicy(env.action_space,model)
learning_algorithm=learners.LearnerPolicyGradient(action_space=env.action_space,average_reward_window=10,torch_model=model,optimizer=optimizer)
learning_algorithm.reset()
while(True):
    learning_algorithm.step(env=env,discount_factor=0.9,maximum_episode_length=100)

    policy=learning_algorithm.get_policy(stochastic=True)
    r=rl_evaluate_policy_multiple_times(env,policy,100,1.0,10)
    print("Evaluation avg reward = %f "% r)

    print("Reward = %f"%learning_algorithm.log.get_last_dynamic_value("total_reward"))







