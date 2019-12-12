import torch_rl.policies as policies
import torch.nn as nn
import torch.optim as optim
import torch_rl.learners as learners
import torch_rl.policies as policies
import torch_rl.environments
import torch_rl.core as core
import gym
import gym.spaces
import torch

LEARNING_RATE=0.001
STDV=0.0000001
NENV=10

envs=[]
for i in range(NENV):
    envs.append(gym.make("Pong-v0"))
    envs[i]=torch_rl.environments.RemapDiscreteEnv(envs[i],[2,3])
#Creation of the policy
#Creation of the policy
A = envs[0].action_space.n

def mapping_function(I):
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    r=torch.Tensor(I.astype(float).ravel())
    return r

for i in range(NENV):
    envs[i]=torch_rl.environments.MappedEnv(envs[i],mapping_function)
N = envs[0].reset().nelement()

print("Number of Actions is: %d" % A)
#===================================================================== CREATION OF THE POLICY =============================================
# Creation of a learning model Q(s): R^N -> R^A
class MyModel(nn.Container):
    def __init__(self, data_size,output_size):
        super(MyModel, self).__init__(
            linear=nn.Linear(data_size, output_size),
            softmax=nn.Softmax()
        )
        self.linear.weight.data.normal_(0,STDV)
        self.linear.bias.data.normal_(0,STDV)
        self.data_size=data_size

    def __call__(self, data):
        output = self.linear(data)
        output=self.softmax(output)
        return output

model = MyModel(N,A)
optimizer= optim.Adam(model.parameters(), lr=LEARNING_RATE)


learning_algorithm=learners.LearnerBatchPolicyGradient(action_space=envs[0].action_space,average_reward_window=10,torch_model=model,optimizer=optimizer)
learning_algorithm.reset()
iteration=0
while(True):
    learning_algorithm.step(envs=envs,discount_factor=0.9,maximum_episode_length=100,render=iteration%10==0)
    print("Reward = %f"%learning_algorithm.log.get_last_dynamic_value("avg_total_reward"))
    iteration=iteration+1







