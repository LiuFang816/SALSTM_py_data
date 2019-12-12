import torch_rl.policies as policies
import torch.nn as nn
import torch.optim as optim
import torch_rl.learners as learners
import torch_rl.environments.control.cartpole as cp
import torch_rl.core as core


LEARNING_RATE=0.01
STDV=0.01
BATCHES=100
envs=[]

print("Creating %d environments" % BATCHES)
for i in range(BATCHES):
    world=cp.World()
    task=cp.Task()
    sensor=cp.Sensor()
    envs.append(core.Env(world,task,sensor))

#Creation of the policy
#Creation of the policy
A = envs[0].action_space.n
N = envs[0].observation_space.high.size
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

    def __call__(self, data):
        output = self.linear(data)
        output=self.tanh(output)
        output=self.linear2(output)
        output=self.softmax(output)
        return output

model = MyModel(N,N*2,A)
optimizer= optim.Adam(model.parameters(), lr=LEARNING_RATE)


learning_algorithm=learners.LearnerBatchPolicyGradient(action_space=envs[0].action_space,average_reward_window=10,torch_model=model,optimizer=optimizer)
learning_algorithm.reset()
while(True):
    learning_algorithm.step(envs=envs,discount_factor=0.9,maximum_episode_length=1000)
    print("Reward = %f (length= %f)"%(learning_algorithm.log.get_last_dynamic_value("avg_total_reward"),learning_algorithm.log.get_last_dynamic_value("avg_length")))







