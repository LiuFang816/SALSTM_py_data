# -*- coding: utf-8 -*-
import numpy as np
import chainer, math, copy, os
from chainer import cuda, Variable, optimizer, optimizers, serializers, function
from chainer import functions as F
from chainer import links as L
from chainer.utils import type_check
from activations import activations
from config import config

class ConvolutionalNetwork(chainer.Chain):
	def __init__(self, **layers):
		super(ConvolutionalNetwork, self).__init__(**layers)
		self.activation_function = "elu"
		self.n_hidden_layers = 0
		self.apply_batchnorm = True
		self.apply_batchnorm_to_input = False

	def forward_one_step(self, x, test):
		f = activations[self.activation_function]
		chain = [x]

		# Hidden convolutinal layers
		for i in range(self.n_hidden_layers):
			u = getattr(self, "layer_%i" % i)(chain[-1])
			if self.apply_batchnorm:
				if i == 0 and self.apply_batchnorm_to_input is False:
					pass
				else:
					u = getattr(self, "batchnorm_%i" % i)(u, test=test)
			chain.append(f(u))

		return chain[-1]

	def __call__(self, x, test=False):
		return self.forward_one_step(x, test=test)

class FullyConnectedNetwork(chainer.Chain):
	def __init__(self, **layers):
		super(FullyConnectedNetwork, self).__init__(**layers)
		self.n_hidden_layers = 0
		self.activation_function = "elu"
		self.apply_batchnorm_to_input = False
		self.projection_type = "fully_connection"
		self.conv_output_filter_size = (1, 1)

	def forward_one_step(self, x, test):
		f = activations[self.activation_function]
		chain = [x]

		# Convert feature maps to a vector
		if self.projection_type == "fully_connection":
			pass

		elif self.projection_type == "global_average_pooling":
			batch_size = chain[-1].data.shape[0]
			n_maps = chain[-1].data[0].shape[0]
			chain.append(F.average_pooling_2d(chain[-1], self.conv_output_filter_size))
			chain.append(F.reshape(chain[-1], (batch_size, n_maps)))

		else:
			raise NotImplementedError()

		# Hidden layers
		for i in range(self.n_hidden_layers):
			u = getattr(self, "layer_%i" % i)(chain[-1])
			if self.apply_batchnorm:
				if i == 0 and self.apply_batchnorm_to_input is False:
					pass
				else:
					u = getattr(self, "batchnorm_%i" % i)(u, test=test)
			output = f(u)
			if self.apply_dropout:
				output = F.dropout(output, train=not test)
			chain.append(output)

		# Output
		u = getattr(self, "layer_%i" % self.n_hidden_layers)(chain[-1])
		if self.apply_batchnorm:
			u = getattr(self, "batchnorm_%i" % self.n_hidden_layers)(u, test=test)
		chain.append(f(u))

		return chain[-1]

	def __call__(self, x, test=False):
		return self.forward_one_step(x, test=test)

		
class Aggregator(function.Function):
	def as_mat(self, x):
		if x.ndim == 2:
			return x
		return x.reshape(len(x), -1)
		
	def check_type_forward(self, in_types):
		n_in = in_types.size()
		type_check.expect(n_in == 3)
		value_type, advantage_type, mean_type = in_types

		type_check.expect(
			value_type.dtype == np.float32,
			advantage_type.dtype == np.float32,
			mean_type.dtype == np.float32,
			value_type.ndim == 2,
			advantage_type.ndim == 2,
			mean_type.ndim == 1,
		)

	def forward(self, inputs):
		value, advantage, mean = inputs
		mean = self.as_mat(mean)
		sub = advantage - mean
		output = value + sub
		return output,

	def backward(self, inputs, grad_outputs):
		xp = cuda.get_array_module(inputs[0])
		gx1 = xp.sum(grad_outputs[0], axis=1)
		gx2 = grad_outputs[0]
		return self.as_mat(gx1), gx2, -gx1

def aggregate(value, advantage, mean):
	return Aggregator()(value, advantage, mean)

class DuelingNetwork:
	def __init__(self):
		print "Initializing DQN..."
		self.exploration_rate = config.rl_initial_exploration

		# Q Network
		self.conv, self.fc_value, self.fc_advantage = build_q_network(config)
		self.load()
		self.update_target()

		# Optimizer
		## RMSProp, ADAM, AdaGrad, AdaDelta, ...
		## See http://docs.chainer.org/en/stable/reference/optimizers.html
		self.optimizer_conv = optimizers.Adam(alpha=config.rl_learning_rate, beta1=config.rl_gradient_momentum)
		self.optimizer_fc_value = optimizers.Adam(alpha=config.rl_learning_rate, beta1=config.rl_gradient_momentum)
		self.optimizer_fc_advantage = optimizers.Adam(alpha=config.rl_learning_rate, beta1=config.rl_gradient_momentum)

		self.optimizer_conv.setup(self.conv)
		self.optimizer_fc_value.setup(self.fc_value)
		self.optimizer_fc_advantage.setup(self.fc_advantage)

		self.optimizer_conv.add_hook(optimizer.GradientClipping(10.0))
		self.optimizer_fc_value.add_hook(optimizer.GradientClipping(10.0))
		self.optimizer_fc_advantage.add_hook(optimizer.GradientClipping(10.0))
		
		# Replay Memory
		## (state, action, reward, next_state, episode_ends)
		shape_state = (config.rl_replay_memory_size, config.rl_agent_history_length * config.ale_screen_channels, config.ale_scaled_screen_size[1], config.ale_scaled_screen_size[0])
		shape_action = (config.rl_replay_memory_size,)
		self.replay_memory = [
			np.zeros(shape_state, dtype=np.float32),
			np.zeros(shape_action, dtype=np.uint8),
			np.zeros(shape_action, dtype=np.int8),
			np.zeros(shape_state, dtype=np.float32),
			np.zeros(shape_action, dtype=np.bool)
		]
		self.total_replay_memory = 0
		self.no_op_count = 0

	def eps_greedy(self, state, exploration_rate):
		prop = np.random.uniform()
		q_max = None
		q_min = None
		if prop < exploration_rate:
			# Select a random action
			action_index = np.random.randint(0, len(config.ale_actions))
		else:
			# Select a greedy action
			state = Variable(state)
			if config.use_gpu:
				state.to_gpu()
			q = self.compute_q_variable(state, test=True)
			if config.use_gpu:
				action_index = cuda.to_cpu(cuda.cupy.argmax(q.data))
				q_max = cuda.to_cpu(cuda.cupy.max(q.data))
				q_min = cuda.to_cpu(cuda.cupy.min(q.data))
			else:
				action_index = np.argmax(q.data)
				q_max = np.max(q.data)
				q_min = np.min(q.data)

		action = self.get_action_with_index(action_index)
		# No-op
		self.no_op_count = self.no_op_count + 1 if action == 0 else 0
		if self.no_op_count > config.rl_no_op_max:
			no_op_index = np.argmin(np.asarray(config.ale_actions))
			actions_without_no_op = []
			for i in range(len(config.ale_actions)):
				if i == no_op_index:
					continue
				actions_without_no_op.append(config.ale_actions[i])
			action_index = np.random.randint(0, len(actions_without_no_op))
			action = actions_without_no_op[action_index]
			print "Reached no_op_max.", "New action:", action

		return action, q_max, q_min

	def store_transition_in_replay_memory(self, state, action, reward, next_state, episode_ends):
		index = self.total_replay_memory % config.rl_replay_memory_size
		self.replay_memory[0][index] = state[0]
		self.replay_memory[1][index] = action
		self.replay_memory[2][index] = reward
		if episode_ends is False:
			self.replay_memory[3][index] = next_state[0]
		self.replay_memory[4][index] = episode_ends
		self.total_replay_memory += 1

	def forward_one_step(self, state, action, reward, next_state, episode_ends, test=False):
		xp = cuda.cupy if config.use_gpu else np
		n_batch = state.shape[0]
		state = Variable(state)
		next_state = Variable(next_state)
		if config.use_gpu:
			state.to_gpu()
			next_state.to_gpu()
		q = self.compute_q_variable(state, test=test)
		q_ = self.compute_q_variable(next_state, test=test)
		max_action_indices = xp.argmax(q_.data, axis=1)
		if config.use_gpu:
			max_action_indices = cuda.to_cpu(max_action_indices)

		# Generate target
		target_q = self.compute_target_q_variable(next_state, test=test)

		# Initialize target signal
		# 教師信号を現在のQ値で初期化
		target = q.data.copy()

		for i in xrange(n_batch):
			# Clip all positive rewards at 1 and all negative rewards at -1
			# プラスの報酬はすべて1にし、マイナスの報酬はすべて-1にする
			if episode_ends[i] is True:
				target_value = np.sign(reward[i])
			else:
				max_action_index = max_action_indices[i]
				target_value = np.sign(reward[i]) + config.rl_discount_factor * target_q.data[i][max_action_indices[i]]
			action_index = self.get_index_with_action(action[i])

			# 現在選択した行動に対してのみ誤差を伝播する。
			# それ以外の行動を表すユニットの2乗誤差は0となる。（target=qとなるため）
			old_value = target[i, action_index]
			diff = target_value - old_value

			# target is an one-hot vector in which the non-zero element(= target signal) corresponds to the taken action.
			# targetは実際にとった行動に対してのみ誤差を考え、それ以外の行動に対しては誤差が0となるone-hotなベクトルです。
			
			# Clip the error to be between -1 and 1.
			# 1を超えるものはすべて1にする。（-1も同様）
			if diff > 1.0:
				target_value = 1.0 + old_value	
			elif diff < -1.0:
				target_value = -1.0 + old_value	
			target[i, action_index] = target_value

		target = Variable(target)

		# Compute error
		loss = F.mean_squared_error(target, q)
		return loss, q

	def replay_experience(self):
		if self.total_replay_memory == 0:
			return
		# Sample random minibatch of transitions from replay memory
		if self.total_replay_memory < config.rl_replay_memory_size:
			replay_index = np.random.randint(0, self.total_replay_memory, (config.rl_minibatch_size, 1))
		else:
			replay_index = np.random.randint(0, config.rl_replay_memory_size, (config.rl_minibatch_size, 1))

		shape_state = (config.rl_minibatch_size, config.rl_agent_history_length * config.ale_screen_channels, config.ale_scaled_screen_size[1], config.ale_scaled_screen_size[0])
		shape_action = (config.rl_minibatch_size,)

		state = np.empty(shape_state, dtype=np.float32)
		action = np.empty(shape_action, dtype=np.uint8)
		reward = np.empty(shape_action, dtype=np.int8)
		next_state = np.empty(shape_state, dtype=np.float32)
		episode_ends = np.empty(shape_action, dtype=np.bool)
		for i in xrange(config.rl_minibatch_size):
			state[i] = self.replay_memory[0][replay_index[i]]
			action[i] = self.replay_memory[1][replay_index[i]]
			reward[i] = self.replay_memory[2][replay_index[i]]
			next_state[i] = self.replay_memory[3][replay_index[i]]
			episode_ends[i] = self.replay_memory[4][replay_index[i]]

		self.optimizer_conv.zero_grads()
		self.optimizer_fc_value.zero_grads()
		self.optimizer_fc_advantage.zero_grads()
		loss, _ = self.forward_one_step(state, action, reward, next_state, episode_ends, test=False)
		loss.backward()
		self.optimizer_fc_value.update()
		self.optimizer_fc_advantage.update()
		self.optimizer_conv.update()

	def compute_q_variable(self, state, test=False):
		output = self.conv(state, test=test)
		value = self.fc_value(output, test=test)
		advantage = self.fc_advantage(output, test=test)
		mean = F.sum(advantage, axis=1) / float(len(config.ale_actions))
		return aggregate(value, advantage, mean)

	def compute_target_q_variable(self, state, test=True):
		output = self.target_conv(state, test=test)
		value = self.target_fc_value(output, test=test)
		advantage = self.target_fc_advantage(output, test=test)
		mean = F.sum(advantage, axis=1) / float(len(config.ale_actions))
		return aggregate(value, advantage, mean)

	def update_target(self):
		self.target_conv = copy.deepcopy(self.conv)
		self.target_fc_value = copy.deepcopy(self.fc_value)
		self.target_fc_advantage = copy.deepcopy(self.fc_advantage)

	def get_action_with_index(self, i):
		return config.ale_actions[i]

	def get_index_with_action(self, action):
		return config.ale_actions.index(action)

	def decrease_exploration_rate(self):
		# Exploration rate is linearly annealed to its final value
		self.exploration_rate -= 1.0 / config.rl_final_exploration_frame
		if self.exploration_rate < config.rl_final_exploration:
			self.exploration_rate = config.rl_final_exploration

	def load(self):
		filename = "conv.model"
		if os.path.isfile(filename):
			serializers.load_hdf5(filename, self.conv)
			print "convolutional network loaded."
		filename = "fc_value.model"
		if os.path.isfile(filename):
			serializers.load_hdf5(filename, self.fc_value)
			print "value network loaded."
		filename = "fc_advantage.model"
		if os.path.isfile(filename):
			serializers.load_hdf5(filename, self.fc_advantage)
			print "action advantage network loaded."

	def save(self):
		serializers.save_hdf5("conv.model", self.conv)
		serializers.save_hdf5("fc_value.model", self.fc_value)
		serializers.save_hdf5("fc_advantage.model", self.fc_advantage)


def build_q_network(config):
	config.check()
	wscale = config.q_wscale

	# Convolutional part of Q-Network
	conv_attributes = {}
	conv_channels = [(config.rl_agent_history_length * config.ale_screen_channels, config.q_conv_hidden_channels[0])]
	conv_channels += zip(config.q_conv_hidden_channels[:-1], config.q_conv_hidden_channels[1:])

	output_map_width = config.ale_scaled_screen_size[0]
	output_map_height = config.ale_scaled_screen_size[1]
	for n in xrange(len(config.q_conv_hidden_channels)):
		output_map_width = (output_map_width - config.q_conv_filter_sizes[n]) / config.q_conv_strides[n] + 1
		output_map_height = (output_map_height - config.q_conv_filter_sizes[n]) / config.q_conv_strides[n] + 1

	for i, (n_in, n_out) in enumerate(conv_channels):
		conv_attributes["layer_%i" % i] = L.Convolution2D(n_in, n_out, config.q_conv_filter_sizes[i], stride=config.q_conv_strides[i], wscale=wscale)
		conv_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)

	conv = ConvolutionalNetwork(**conv_attributes)
	conv.n_hidden_layers = len(config.q_conv_hidden_channels)
	conv.activation_function = config.q_conv_activation_function
	conv.apply_batchnorm = config.apply_batchnorm
	conv.apply_batchnorm_to_input = config.q_conv_apply_batchnorm_to_input
	if config.use_gpu:
		conv.to_gpu()

	# Value network
	fc_attributes = {}
	if config.q_conv_output_projection_type == "fully_connection":
		fc_units = [(output_map_width * output_map_height * config.q_conv_hidden_channels[-1], config.q_fc_hidden_units[0])]
		fc_units += [(config.q_fc_hidden_units[0], config.q_fc_hidden_units[1])]
	elif config.q_conv_output_projection_type == "global_average_pooling":
		fc_units = [(config.q_conv_hidden_channels[-1], config.q_fc_hidden_units[0])]
		fc_units += [(config.q_fc_hidden_units[0], config.q_fc_hidden_units[1])]
	else:
		raise NotImplementedError()
	fc_units += zip(config.q_fc_hidden_units[1:-1], config.q_fc_hidden_units[2:])
	fc_units += [(config.q_fc_hidden_units[-1], 1)]

	for i, (n_in, n_out) in enumerate(fc_units):
		fc_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
		fc_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)

	fc_value = FullyConnectedNetwork(**fc_attributes)
	fc_value.n_hidden_layers = len(fc_units) - 1
	fc_value.activation_function = config.q_fc_activation_function
	fc_value.apply_batchnorm = config.apply_batchnorm
	fc_value.apply_dropout = config.q_fc_apply_dropout
	fc_value.apply_batchnorm_to_input = config.q_fc_apply_batchnorm_to_input
	fc_value.conv_output_filter_size = (output_map_width, output_map_height)
	fc_value.projection_type = config.q_conv_output_projection_type
	if config.use_gpu:
		fc_value.to_gpu()

	# Action advantage network
	fc_attributes = {}
	if config.q_conv_output_projection_type == "fully_connection":
		fc_units = [(output_map_width * output_map_height * config.q_conv_hidden_channels[-1], config.q_fc_hidden_units[0])]
		fc_units += [(config.q_fc_hidden_units[0], config.q_fc_hidden_units[1])]
	elif config.q_conv_output_projection_type == "global_average_pooling":
		fc_units = [(config.q_conv_hidden_channels[-1], config.q_fc_hidden_units[0])]
		fc_units += [(config.q_fc_hidden_units[0], config.q_fc_hidden_units[1])]
	else:
		raise NotImplementedError()
	fc_units += zip(config.q_fc_hidden_units[1:-1], config.q_fc_hidden_units[2:])
	fc_units += [(config.q_fc_hidden_units[-1], len(config.ale_actions))]
	for i, (n_in, n_out) in enumerate(fc_units):
		fc_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
		fc_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)

	fc_advantage = FullyConnectedNetwork(**fc_attributes)
	fc_advantage.n_hidden_layers = len(fc_units) - 1
	fc_advantage.activation_function = config.q_fc_activation_function
	fc_advantage.apply_batchnorm = config.apply_batchnorm
	fc_advantage.apply_dropout = config.q_fc_apply_dropout
	fc_advantage.apply_batchnorm_to_input = config.q_fc_apply_batchnorm_to_input
	fc_advantage.conv_output_filter_size = (output_map_width, output_map_height)
	fc_advantage.projection_type = config.q_conv_output_projection_type
	if config.use_gpu:
		fc_advantage.to_gpu()

	return conv, fc_value, fc_advantage