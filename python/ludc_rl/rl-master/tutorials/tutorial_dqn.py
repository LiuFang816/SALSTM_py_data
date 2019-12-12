import torch_rl.policies as policies
import torch.nn as nn
import torch.optim as optim
import torch_rl.learners as learners
import gym


LEARNING_RATE=0.001
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
        self.linear.weight.data.normal_(0,STDV)
        self.linear.bias.data.normal_(0,STDV)
        self.linear2.weight.data.normal_(0,STDV)
        self.linear2.bias.data.normal_(0,STDV)
        self.data_size=data_size

    def __call__(self, data):
        output = self.linear(data)
        output=self.tanh(output)
        output=self.linear2(output)
        return output

model = MyModel(N,N*2,A)
optimizer= optim.Adam(model.parameters(), lr=LEARNING_RATE)

greedy_policy=policies.DiscreteModelPolicy(env.action_space,model)
exploration_policy=policies.DiscreteEpsilonGreedyPolicy(env.action_space,0.1,greedy_policy)

learning_algorithm=learners.LearnerDQN(action_space=env.action_space,observation_space=env.observation_space,size_batch_replay_memory=10,nb_batch_replay_memory=100,exploration_policy=exploration_policy,torch_model=model,optimizer=optimizer)
learning_algorithm.reset()
while(True):
    learning_algorithm.step(env=env,discount_factor=0.8,maximum_episode_length=30)
    print("Reward = %f"%learning_algorithm.log.get_last_dynamic_value("total_reward"))







