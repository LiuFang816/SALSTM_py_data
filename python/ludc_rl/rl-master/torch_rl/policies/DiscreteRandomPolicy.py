from torch_rl.policies.Policy import Policy

class DiscreteRandomPolicy(Policy):
	def __init__(self,action_space):
		Policy.__init__(self, action_space)

    #Sample an action
	def sample(self):
		return self.action_space.sample()

