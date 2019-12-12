# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
import scipy.misc as spm
from rlglue.agent import AgentLoader
sys.path.append(os.path.split(os.getcwd())[0])
from PIL import Image
from config import config
from agent import Agent

# Override config
config.ale_actions = [4, 3, 1, 0]
config.ale_screen_size = [210, 160]
config.ale_scaled_screen_size = [84, 84]
config.rl_replay_memory_size = 10 ** 5
config.rl_replay_start_size = 10 ** 4
config.q_conv_hidden_channels = [32, 64, 64]
config.q_conv_strides = [4, 2, 1]
config.q_conv_filter_sizes = [8, 4, 3]
config.q_fc_hidden_units = [512, 128]
config.rl_update_frequency = 1

# Override agent
class BreakoutAgent(Agent):
	pass

AgentLoader.loadAgent(BreakoutAgent())
