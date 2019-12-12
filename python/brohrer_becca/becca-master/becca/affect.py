"""
The Affect class.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import becca.tools as tools


class Affect(object):
    """
    Track reward over time.

    Affect, or mood, is the level of arousal of the brain.
    It is influenced by the recent history of reward and in turn
    influences the intensity with which a brain pursues
    its goals and makes plans.
    """
    def __init__(self):
        """
        Set up Affect.
        """
        # satisfaction_time_constant : float
        #     The time constant of the leaky integrator used to filter
        #     reward into a rough average of the recent reward history.
        self.satisfaction_time_constant = 1e3
        # satisfaction : float
        #     A filtered average of the reward.
        self.satisfaction = 0.
        # cumulative_reward : float
        #     The total reward amassed since the last visualization.
        self.cumulative_reward = 0.
        # time_since_reward_log : int
        #     Number of time steps since reward was last logged. It gets
        #     logged every time Affect is visualized.
        self.time_since_reward_log = 0.
        # reward_history : list of floats
        #     A time series of reward accumulated during the periods between
        #     each time Affect is visualized.
        self.reward_history = []
        # reward_steps : list of ints
        #     A time series of the brain's age in time steps corresponding
        #     to each of the rewards in reward_history.
        self.reward_steps = []


    def update(self, reward):
        """
        Update the current level of satisfaction and record the reward.

        Parameters
        ----------
        reward : float
            The most recently observed reward value.

        Returns
        -------
        self.satisfaction : float
        """
        # Clip the reward so that it falls between -1 and 1.
        reward = np.maximum(np.minimum(reward, 1.), -1.)

        # Update the satisfaction, a filtered version of the reward.
        rate = 1. / self.satisfaction_time_constant
        # This filter is also known as a leaky integrator.
        self.satisfaction = self.satisfaction * (1. - rate) + reward * rate

        # Log the reward.
        self.cumulative_reward += reward
        self.time_since_reward_log += 1

        return self.satisfaction


    def visualize(self, brain):
        """
        Update the reward history, create plots, and save them to a file.

        Parameters
        ----------
        brain : Brain
            The brain to which the affect belongs.

        Returns
        -------
        performance : float
            The average reward over the lifespan of the brain.
        """
        # Check whether any time has passed since the last update.
        if self.time_since_reward_log > 0:
            # Update the lifetime record of the reward.
            self.reward_history.append(float(self.cumulative_reward) /
                                       float(self.time_since_reward_log))
            self.cumulative_reward = 0
            self.time_since_reward_log = 0
            self.reward_steps.append(brain.timestep)

        performance = np.mean(self.reward_history)

        # Plot the lifetime record of the reward.
        fig = plt.figure(11111)
        color = (np.array(tools.copper) +
                 np.random.normal(size=3, scale=.1))
        color = np.maximum(np.minimum(color, 1.), 0.)
        color = tuple(color)
        linewidth = np.random.normal(loc=2.5)
        linewidth = 2
        linewidth = np.maximum(1., linewidth)
        plt.plot(
            np.array(self.reward_steps) / 1000.,
            self.reward_history,
            color=color,
            linewidth=linewidth)
        plt.gca().set_axis_bgcolor(tools.copper_highlight)
        plt.xlabel('Thousands of time steps')
        plt.ylabel('Average reward')
        plt.title('Reward history for {0}'.format(brain.name))
        fig.show()
        fig.canvas.draw()

        # Save a copy of the plot.
        filename = 'reward_history_{0}.png'.format(brain.name)
        pathname = os.path.join(brain.log_dir, filename)
        plt.savefig(pathname, format='png')

        return performance
