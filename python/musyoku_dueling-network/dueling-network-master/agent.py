# -*- coding: utf-8 -*-
import copy
import scipy.misc as spm
import numpy as np
from rlglue.agent.Agent import Agent as RLGlueAgent
from rlglue.types import Action
from rlglue.utils import TaskSpecVRLGLUE3
from dueling_network import DuelingNetwork
from config import config
from PIL import Image

class Agent(RLGlueAgent):
	def __init__(self):
		self.last_action = 0
		self.time_step = 0
		self.total_time_step = 0
		self.episode_step = 0
		self.populating_phase = False

		self.model_save_interval = 30

		# Switch learning phase / evaluation phase
		self.policy_frozen = False

		self.duel = DuelingNetwork()
		self.state = np.zeros((config.rl_agent_history_length, config.ale_screen_channels, config.ale_scaled_screen_size[1], config.ale_scaled_screen_size[0]), dtype=np.float32)
		self.exploration_rate = self.duel.exploration_rate
		self.exploration_rate_for_evaluation = 0.05
		self.last_observed_screen = None

	def preprocess_screen(self, observation):
		screen_width = config.ale_screen_size[0]
		screen_height = config.ale_screen_size[1]
		new_width = config.ale_scaled_screen_size[0]
		new_height = config.ale_scaled_screen_size[1]
		if len(observation.intArray) == 100928: 
			observation = np.asarray(observation.intArray[128:], dtype=np.uint8).reshape((screen_width, screen_height, 3))
			observation = spm.imresize(observation, (new_height, new_width))
			# Clip the pixel value to be between 0 and 1
			if config.ale_screen_channels == 1:
				# Convert RGB to Luminance
				observation = np.dot(observation[:,:,:], [0.299, 0.587, 0.114])
				observation = observation.reshape((new_height, new_width, 1))
			observation = observation.transpose(2, 0, 1) / 255.0
			observation /= (np.max(observation) + 1e-5)
		else:
			# Greyscale
			if config.ale_screen_channels == 3:
				raise Exception("You forgot to add --send_rgb option when you run ALE.")
			observation = np.asarray(observation.intArray[128:]).reshape((screen_width, screen_height))
			observation = spm.imresize(observation, (new_height, new_width))
			# Clip the pixel value to be between 0 and 1
			observation = observation.reshape((1, new_height, new_width)) / 255.0
			observation /= (np.max(observation) + 1e-5)

		observed_screen = observation
		if self.last_observed_screen is not None:
			observed_screen = np.maximum(observation, self.last_observed_screen)

		self.last_observed_screen = observation
		return observed_screen

	def agent_init(self, taskSpecString):
		pass

	def reshape_state_to_conv_input(self, state):
		return state.reshape((1, config.rl_agent_history_length * config.ale_screen_channels, config.ale_scaled_screen_size[1], config.ale_scaled_screen_size[0]))

	def dump_result(self, reward, q_max=None, q_min=None):
		if self.time_step % 50 == 0:
			if self.policy_frozen is False:
				print "time_step:", self.time_step,
				
			print "reward:", reward,
			print "eps:", self.exploration_rate,
			if q_min is None:
				print ""
			else:
				print "Q ::",
				print "max:", q_max,
				print "min:", q_min

	def dump_state(self, state=None, prefix=""):
		if state is None:
			state = self.state
		state = self.reshape_state_to_conv_input(state)
		for h in xrange(config.rl_agent_history_length):
			start = h * config.ale_screen_channels
			end = start + config.ale_screen_channels
			image = state[0,start:end,:,:]
			if config.ale_screen_channels == 1:
				image = image.reshape((image.shape[1], image.shape[2]))
			elif config.ale_screen_channels == 3:
				image = image.transpose(1, 2, 0)
			image = np.uint8(image * 255.0)
			image = Image.fromarray(image)
			image.save(("%sstate-%d.png" % (prefix, h)))

	def learn(self, reward, epsode_ends=False):
		if self.policy_frozen is False:
			self.duel.store_transition_in_replay_memory(self.reshape_state_to_conv_input(self.last_state), self.last_action, reward, self.reshape_state_to_conv_input(self.state), epsode_ends)
			if self.total_time_step <= config.rl_replay_start_size:
				# A uniform random policy is run for 'replay_start_size' frames before learning starts
				# 経験を積むためランダムに動き回るらしい。
				print "Initial exploration before learning starts:", "%d/%d" % (self.total_time_step, config.rl_replay_start_size)
				self.populating_phase = True
				self.exploration_rate = config.rl_initial_exploration
			else:
				self.populating_phase = False
				self.duel.decrease_exploration_rate()
				self.exploration_rate = self.duel.exploration_rate

				if self.total_time_step % (config.rl_action_repeat * config.rl_update_frequency) == 0 and self.total_time_step != 0:
					self.duel.replay_experience()

				if self.total_time_step % config.rl_target_network_update_frequency == 0 and self.total_time_step != 0:
					print "Target has been updated."
					self.duel.update_target()

	def agent_start(self, observation):
		print "Episode", self.episode_step, "::", "total_time_step:",
		if self.total_time_step > 1000:
			print int(self.total_time_step / 1000), "K"
		else:
			print self.total_time_step
		observed_screen = self.preprocess_screen(observation)
		self.state[0] = observed_screen

		return_action = Action()
		action, q_max, q_min = self.duel.eps_greedy(self.reshape_state_to_conv_input(self.state), self.exploration_rate)
		return_action.intArray = [action]

		self.last_action = action
		self.last_state = self.state

		return return_action

	def agent_step(self, reward, observation):
		observed_screen = self.preprocess_screen(observation)
		self.state = np.roll(self.state, 1, axis=0)
		self.state[0] = observed_screen

		########################### DEBUG ###############################
		# if self.total_time_step % 500 == 0 and self.total_time_step != 0:
		# 	self.dump_state()


		self.learn(reward)
		
		return_action = Action()
		q_max = None
		q_min = None
		if self.time_step % config.rl_action_repeat == 0:
			action, q_max, q_min = self.duel.eps_greedy(self.reshape_state_to_conv_input(self.state), self.exploration_rate)
		else:
			action = self.last_action
		return_action.intArray = [action]
		self.last_action = action

		self.dump_result(reward, q_max, q_min)

		if self.policy_frozen is False:
			self.last_state = self.state
			self.time_step += 1
			self.total_time_step += 1

		return return_action

	def agent_end(self, reward):
		self.learn(reward, epsode_ends=True)

		# [Optional]
		## Visualizing the results
		self.dump_result(reward)

		if self.policy_frozen is False:
			self.time_step = 0
			self.total_time_step += 1
			self.episode_step += 1

	def agent_cleanup(self):
		pass

	def agent_message(self, inMessage):
		if inMessage.startswith("freeze_policy"):
			self.policy_frozen = True
			self.exploration_rate = self.exploration_rate_for_evaluation
			return "The policy was freezed."

		if inMessage.startswith("unfreeze_policy"):
			self.policy_frozen = False
			self.exploration_rate = self.duel.exploration_rate
			return "The policy was unfreezed."

		if inMessage.startswith("save_model"):
			if self.populating_phase is False:
				self.duel.save()
			return "The model was saved."