# -*- coding: utf-8 -*-
import math, pylab, time, argparse, os
import pandas as pd
import numpy as np
import seaborn as sns
import rlglue.RLGlue as RLGlue
from config import config

max_episode = 10 ** 5
total_episode = 0
learned_episode = 0
learned_steps = 0
total_time = 0
time_steps_per_epoch = 5 * 10 ** 4
highscore = 0
saving_freq = 100

# evaluation
num_episode_between_evaluations = 100
is_evaluation_phase = False
num_finished_eval_episode = 0
sum_evaluation_scores = 0
num_episode_per_evaluation = 20
evaluation_scores = np.zeros((num_episode_per_evaluation,), dtype=np.float64)

# args
parser = argparse.ArgumentParser()
parser.add_argument("--csv_dir", type=str, default="csv")
parser.add_argument("--plot_dir", type=str, default="plot")
args = parser.parse_args()

try:
	os.mkdir(args.plot_dir)
	os.mkdir(args.csv_dir)
except:
	pass

# csv
csv_writing_freq = 100
csv_episode = []
csv_training_highscore = []
csv_evaluation = []

# plot
plot_freq = csv_writing_freq
sns.set_style("ticks")
sns.set_style("whitegrid", {"grid.linestyle": "--"})
sns.set_context("poster")

def plot_episode_reward():
	pylab.clf()
	sns.set_context("poster")
	pylab.plot(0, 0)
	episodes = [0]
	scores = [0]
	for n in xrange(len(csv_episode)):
		params = csv_episode[n]
		episodes.append(params[0])
		scores.append(params[1])
	pylab.plot(episodes, scores, sns.xkcd_rgb["windows blue"], lw=2)
	pylab.xlabel("episodes")
	pylab.ylabel("score")
	pylab.savefig("%s/episode_reward.png" % args.plot_dir)

def plot_evaluation_episode_reward():
	pylab.clf()
	sns.set_context("poster")
	pylab.plot(0, 0)
	episodes = [0]
	average_scores = [0]
	median_scores = [0]
	for n in xrange(len(csv_evaluation)):
		params = csv_evaluation[n]
		episodes.append(params[0])
		average_scores.append(params[1])
		median_scores.append(params[2])
	pylab.plot(episodes, average_scores, sns.xkcd_rgb["windows blue"], lw=2)
	pylab.xlabel("episodes")
	pylab.ylabel("average score")
	pylab.savefig("%s/evaluation_episode_average_reward.png" % args.plot_dir)

	pylab.clf()
	pylab.plot(0, 0)
	pylab.plot(episodes, median_scores, sns.xkcd_rgb["windows blue"], lw=2)
	pylab.xlabel("episodes")
	pylab.ylabel("median score")
	pylab.savefig("%s/evaluation_episode_median_reward.png" % args.plot_dir)

def plot_training_episode_highscore():
	pylab.clf()
	sns.set_context("poster")
	pylab.plot(0, 0)
	episodes = [0]
	highscore = [0]
	for n in xrange(len(csv_training_highscore)):
		params = csv_training_highscore[n]
		episodes.append(params[0])
		highscore.append(params[1])
	pylab.plot(episodes, highscore, sns.xkcd_rgb["windows blue"], lw=2)
	pylab.xlabel("episodes")
	pylab.ylabel("highscore")
	pylab.savefig("%s/training_episode_highscore.png" % args.plot_dir)


def run_episode(training=True):
	global total_episode, learned_episode, total_time, learned_steps, csv_episode, highscore, num_finished_eval_episode, evaluation_scores
	start_time = time.time()
	RLGlue.RL_episode(0)
	num_steps = RLGlue.RL_num_steps()
	total_reward = RLGlue.RL_return()
	total_episode += 1
	elapsed_time = time.time() - start_time
	total_time += elapsed_time
	epoch = int(learned_steps / time_steps_per_epoch)

	if training:
		learned_steps += num_steps
		learned_episode += 1
		sec = int(elapsed_time)
		total_minutes = int(total_time / 60)
		csv_episode.append([learned_episode, total_reward, num_steps, sec, total_minutes, epoch, learned_steps])
		if total_reward > highscore:
			highscore = total_reward
			csv_training_highscore.append([learned_episode, highscore, total_minutes, epoch])
		print "Episode:", learned_episode, "epoch:", epoch, "num_steps:", num_steps, "total_reward:", total_reward, "time:", sec, "sec",  "total_time:", total_minutes, "min", "lr:", config.rl_learning_rate

	return num_steps, total_reward


RLGlue.RL_init()

while learned_episode < max_episode:
	epoch = int(learned_steps / time_steps_per_epoch)
	total_minutes = int(total_time / 60)

	if learned_episode % num_episode_between_evaluations == 0 and total_episode != 0:
		if is_evaluation_phase is False:
			print "Freezing the policy for evaluation."
			RLGlue.RL_agent_message("freeze_policy")
			num_finished_eval_episode = 0
			is_evaluation_phase = True
		num_steps, total_reward = run_episode(training=False)
		evaluation_scores[num_finished_eval_episode] = total_reward
		num_finished_eval_episode += 1
		print "Evaluation (", num_finished_eval_episode, "/" , num_episode_per_evaluation, ") ::", "num_steps:", num_steps, "total_reward:", total_reward
		if num_finished_eval_episode == num_episode_per_evaluation:
			is_evaluation_phase = False
			csv_evaluation.append([learned_episode, np.mean(evaluation_scores), np.median(evaluation_scores), total_minutes, epoch])
			RLGlue.RL_agent_message("unfreeze_policy")
		else:
			continue

	if learned_episode % saving_freq == 0 and learned_episode != 0:
		print "Saving the model."
		RLGlue.RL_agent_message("save_model")

	if learned_episode % csv_writing_freq == 0 and learned_episode != 0:
		print "Writing to csv files."
		if len(csv_episode):
			data = pd.DataFrame(csv_episode)
			data.columns = ["episode", "reward", "num_steps", "sec", "total_minutes", "epoch", "total_steps"]
			data.to_csv("%s/episode.csv" % args.csv_dir)

		if len(csv_training_highscore):
			data = pd.DataFrame(csv_training_highscore)
			data.columns = ["episode", "highscore", "total_minutes", "epoch"]
			data.to_csv("%s/training_highscore.csv" % args.csv_dir)

		if len(csv_evaluation) > 0:
			data = pd.DataFrame(csv_evaluation)
			data.columns = ["episode", "average", "median", "total_minutes", "epoch"]
			data.to_csv("%s/evaluation.csv" % args.csv_dir)

	if learned_episode % plot_freq == 0 and learned_episode != 0:
		print "Plotting the csv data."
		plot_episode_reward()
		plot_training_episode_highscore()
		plot_evaluation_episode_reward()

	run_episode(training=True)

RLGlue.RL_cleanup()

print "Experiment has ended at episode", total_episode