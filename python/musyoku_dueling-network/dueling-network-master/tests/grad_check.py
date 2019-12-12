# -*- coding: utf-8 -*-
import sys, os
from chainer import cuda, optimizers, gradient_check, Variable
sys.path.append(os.path.split(os.getcwd())[0])
from dueling_network import *
from config import config

# Override config
config.ale_actions = [4, 3, 1, 0]
config.ale_screen_size = [210, 160]
config.ale_scaled_screen_size = [84, 84]
config.rl_replay_memory_size = 10 ** 5
config.rl_replay_start_size = 10 ** 4
config.q_conv_hidden_channels = [32, 64, 64]
config.q_conv_strides = [4, 2, 1]
config.q_conv_filter_sizes = [8, 4, 3]
config.q_fc_hidden_units = [128, 64]
config.apply_batchnorm = True
config.use_gpu = False
config.q_conv_output_projection_type = "global_average_pooling"


def backprop_check():
	xp = cuda.cupy if config.use_gpu else np
	duel = DuelingNetwork()

	state = xp.random.uniform(-1.0, 1.0, (2, config.rl_agent_history_length * config.ale_screen_channels, config.ale_scaled_screen_size[1], config.ale_scaled_screen_size[0])).astype(xp.float32)
	reward = [1, 0]
	action = [3, 4]
	episode_ends = [0, 0]
	next_state = xp.random.uniform(-1.0, 1.0, (2, config.rl_agent_history_length * config.ale_screen_channels, config.ale_scaled_screen_size[1], config.ale_scaled_screen_size[0])).astype(xp.float32)

	optimizer_conv = optimizers.Adam(alpha=config.rl_learning_rate, beta1=config.rl_gradient_momentum)
	optimizer_conv.setup(duel.conv)
	optimizer_fc_value = optimizers.Adam(alpha=config.rl_learning_rate, beta1=config.rl_gradient_momentum)
	optimizer_fc_value.setup(duel.fc_value)
	optimizer_fc_advantage = optimizers.Adam(alpha=config.rl_learning_rate, beta1=config.rl_gradient_momentum)
	optimizer_fc_advantage.setup(duel.fc_advantage)

	for i in xrange(10000):
		optimizer_conv.zero_grads()
		optimizer_fc_value.zero_grads()
		optimizer_fc_advantage.zero_grads()
		loss, _ = duel.forward_one_step(state, action, reward, next_state, episode_ends)
		loss.backward()
		optimizer_conv.update()
		optimizer_fc_value.update()
		optimizer_fc_advantage.update()
		print loss.data,
		print duel.conv.layer_2.W.data[0, 0, 0, 0],
		print duel.fc_value.layer_2.W.data[0, 0],
		print duel.fc_advantage.layer_2.W.data[0, 0]

def grad_check():
	xp = cuda.cupy if config.use_gpu else np
	value = Variable(xp.random.uniform(-1.0, 1.0, (2, 1)).astype(xp.float32))
	advantage = Variable(xp.random.uniform(-1.0, 1.0, (2, 5)).astype(xp.float32))
	mean = Variable(xp.random.uniform(-1.0, 1.0, (2,)).astype(xp.float32))
	q = aggregate(value, advantage, mean)
	gy = xp
	print q.data
	for i in xrange(2):
		for m in xrange(5):
			print value.data[i] + advantage.data[i, m] - mean.data[i]

	y_grad = xp.ones((2, 5)).astype(xp.float32)
	gradient_check.check_backward(Aggregator(), (value.data, advantage.data, mean.data), y_grad, eps=1e-2)

backprop_check()
grad_check()
	
