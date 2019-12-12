# -*- coding: utf-8 -*-
import math
from activations import activations

class Config:
	def check(self):
		if self.ale_screen_channels != 1 and self.ale_screen_channels != 3:
			raise Exception("Invalid channels for ale_screen_channels.")

		if self.q_conv_activation_function not in activations:
			raise Exception("Invalid activation function for q_conv_activation_function.")
		if self.q_fc_activation_function not in activations:
			raise Exception("Invalid activation function for q_fc_activation_function.")

		n_conv_hidden_layers = len(self.q_conv_hidden_channels)
		if len(self.q_conv_filter_sizes) != n_conv_hidden_layers:
			raise Exception("Invlaid number of elements for q_conv_filter_sizes")
		if len(self.q_conv_strides) != n_conv_hidden_layers:
			raise Exception("Invlaid number of elements for q_conv_strides")

		q_output_map_width = self.ale_scaled_screen_size[0]
		q_output_map_height = self.ale_scaled_screen_size[1]
		stndrdth = ("st", "nd", "rd")
		for n in xrange(len(self.q_conv_hidden_channels)):
			if (q_output_map_width - self.q_conv_filter_sizes[n]) % self.q_conv_strides[n] != 0:
				print "WARNING"
				print "at", (("%d%s" % (n + 1, stndrdth[n])) if n < 3 else ("%dth" % (n + 1))), "conv layer:"
				print "width of input maps:", q_output_map_width
				print "stride:", self.q_conv_strides[n]
				print "filter size:", (self.q_conv_filter_sizes[n], self.q_conv_filter_sizes[n])
				print "width of outout maps:", (q_output_map_width - self.q_conv_filter_sizes[n]) / float(self.q_conv_strides[n]) + 1
				print "The width of output maps MUST be an integer!"
				possible_strides = []
				for _stride in range(1, 11):
					if (q_output_map_width - self.q_conv_filter_sizes[n]) % _stride == 0:
						possible_strides.append(_stride)
				if len(possible_strides) > 0:
					print "I recommend you to"
					print "	use stride of", possible_strides
					print "	or"
				print "	change input image size to",
				new_image_width = int(math.ceil((q_output_map_width - self.q_conv_filter_sizes[n]) / float(self.q_conv_strides[n]))) * self.q_conv_strides[n] + self.q_conv_filter_sizes[n]
				new_image_height = q_output_map_height
				for _n in xrange(n):
					new_image_width = (new_image_width - 1) * self.q_conv_strides[_n] + self.q_conv_filter_sizes[_n]
					new_image_height = (new_image_height - 1) * self.q_conv_strides[_n] + self.q_conv_filter_sizes[_n]
				print (new_image_width, new_image_height)
				raise Exception()
			if q_output_map_height % self.q_conv_strides[n] != 0:
				print "WARNING"
				print "at", (("%d%s" % (n + 1, stndrdth[n])) if n < 3 else ("%dth" % (n + 1))), "conv layer:"
				print "height of input maps:", q_output_map_height
				print "stride:", self.q_conv_strides[n]
				print "filter size:", (self.q_conv_filter_sizes[n], self.q_conv_filter_sizes[n])
				print "height of outout maps:", (q_output_map_height - self.q_conv_filter_sizes[n]) / float(self.q_conv_strides[n]) + 1
				print "The height of output maps MUST be an integer!"
				possible_strides = []
				for _stride in range(1, 11):
					if (q_output_map_height - self.q_conv_filter_sizes[n]) % _stride == 0:
						possible_strides.append(_stride)
				if len(possible_strides) > 0:
					print "I recommend you to"
					print "	use stride of", possible_strides
					print "	or"
				print "	change input image size to",
				new_image_width = q_output_map_width
				new_image_height = int(math.ceil((q_output_map_height - self.q_conv_filter_sizes[n]) / float(self.q_conv_strides[n]))) * self.q_conv_strides[n] + self.q_conv_filter_sizes[n]
				for _n in xrange(n):
					new_image_width = (new_image_width - 1) * self.q_conv_strides[_n] + self.q_conv_filter_sizes[_n]
					new_image_height = (new_image_height - 1) * self.q_conv_strides[_n] + self.q_conv_filter_sizes[_n]
				print (new_image_width, new_image_height)
				raise Exception()
			q_output_map_width = (q_output_map_width - self.q_conv_filter_sizes[n]) / self.q_conv_strides[n] + 1
			q_output_map_height = (q_output_map_height - self.q_conv_filter_sizes[n]) / self.q_conv_strides[n] + 1
		if q_output_map_width <= 0 or q_output_map_height <= 0:
			raise Exception("The size of the output feature maps will be 0 in the current settings.")

		if self.q_conv_output_projection_type not in {"fully_connection", "global_average_pooling"}:
			raise Exception("Invalid type of projection for q_conv_output_projection_type.")

		if len(self.q_fc_hidden_units) == 0:
			raise Exception("You must add at least one layer to fully connected network.")
		if self.rl_replay_start_size > self.rl_replay_memory_size:
			self.rl_replay_start_size = self.rl_replay_memory_size
		if self.rl_action_repeat < 1:
			self.rl_action_repeat = 1


config = Config()

# General
config.use_gpu = True
config.apply_batchnorm = True

# ALE
## Raw screen image width and height.
## [width, height]
config.ale_screen_size = [120, 280]

## Scaled screen image width and height.
## Input scaled images to convolutional network
## [width, height]
config.ale_scaled_screen_size = [62, 142]

## greyscale -> 1
## rgb -> 3
config.ale_screen_channels = 1

## List of actions
## The required actions are written in ale_dir/src/games/supported/the_name_of_the_game_you_want_to_play.cpp,
## The corrensponding integers are defined in ale_dir/src/common/Constants.h 
## ゲームをプレイするのに必要な操作のリスト。
## それぞれのゲームでどの操作が必要かはale_dir/src/games/supported/ゲーム名.cppに書いてあります。
## 各操作の定義はale_dir/src/common/Constants.hで行われているので参照し、数値に変換してください。
config.ale_actions = [0, 3, 4]

# Reinforcment Learning
## These hyperparameters are based on the original Nature paper.
## For more details see following:
## [Human-level control through deep reinforcement learning](http://www.nature.com/nature/journal/v518/n7540/abs/nature14236.html)

## The number of most recent frames experienced by the agent that are given as input to the convolutional network
config.rl_agent_history_length = 4

## This value must be 1 or more 
## Original Nature paper: 4 (= config.rl_action_repeat * --frame_skip(We use a value of 4 when we run the ALE))
## この値を大きくするとエージェントはゆったり動くようになります
config.rl_action_repeat = 1

config.rl_minibatch_size = 32
config.rl_replay_memory_size = 10 ** 6
config.rl_target_network_update_frequency = 10 ** 4
config.rl_discount_factor = 0.99
config.rl_update_frequency = 4
config.rl_learning_rate = 0.00025
config.rl_gradient_momentum = 0.95
config.rl_initial_exploration = 1.0
config.rl_final_exploration = 0.1
config.rl_final_exploration_frame = 10 ** 6
config.rl_replay_start_size = 5 * 10 ** 4
config.rl_no_op_max = 30

# Q-Network
## The list of the number of channels for each hidden convolutional layer (input side -> output side)
## Q関数を構成する畳み込みネットワークの隠れ層のチャネル数。
config.q_conv_hidden_channels = [32, 64, 64]

## The list of stride for each hidden convolutional layer (input side -> output side)
config.q_conv_strides = [4, 2, 1]

## The list of filter size of each convolutional layer (input side -> output side)
config.q_conv_filter_sizes = [8, 4, 3]

## See activations.py
config.q_conv_activation_function = "elu"

## Whether or not to apply batch normalization to the input of convolutional network (the raw screen image from ALE)
## This overrides config.apply_batchnorm
## 畳み込み層への入力（つまりゲーム画面の画像データ）にバッチ正規化を適用するかどうか
## config.apply_batchnormの設定によらずこちらが優先されます
config.q_conv_apply_batchnorm_to_input = True

## The number of units for each fully connected layer.
## These are placed on top of the convolutional network.
## 畳み込み層を接続する全結合層のユニット数を入力側から出力側に向かって並べてください。
config.q_fc_hidden_units = [256, 128]

## "global_average_pooling" or "fully_connection"
## Specify how to convert the output feature maps to vector
## For more details on Global Average Pooling, see following papers:
## Network in Network(http://arxiv.org/abs/1312.440)0
config.q_conv_output_projection_type = "fully_connection"

## See activations.py
config.q_fc_activation_function = "elu"

## Whether or not to apply dropout to all fully connected layers
config.q_fc_apply_dropout = False

## Whether or not to apply batch normalization to the input of fully connected network (the output of convolutional network)
## This overrides config.apply_batchnorm
## 全結合層への入力（つまり畳み込み層の出力）にバッチ正規化を適用するかどうか
## config.apply_batchnormの設定によらずこちらが優先されます
config.q_fc_apply_batchnorm_to_input = True

## Default: 1.0
config.q_wscale = 1.0