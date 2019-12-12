"""
The Model class.
"""

from __future__ import print_function
import os

import numpy as np
#import matplotlib.patches as patches
#import matplotlib.pyplot as plt

import becca.model_numba as nb
import becca.model_viz as viz


class Model(object):
    """
    Build a predictive model based on sequences of features, goals and reward.

    This version of Becca is model-based, meaning that it builds a
    predictive model of its world in the form of a set of sequences.
    It builds prefixes of the form feature-goal and associates a reward
    and curiosity with each. This is similar to the state-action
    value functions of Q-learning.

    The model also builds feature-goal-feature sequences. These are
    similar to state-action-reward-state-action (SARSA) tuples, as in
    Online Q-Learning using Connectionist Systems" Rummery & Niranjan (1994))
    This formulation allows for prediction, action selection and planning.

    Prediction.
    Knowing the current active features
    and recent goals, both the reward and the resulting features can be
    anticipated.

    Action selection.
    Knowing the
    current active features, goals can be chosen in order to reach
    a desired feature or to maximize reward.

    Planning.
    Feature-goal-feature tuples can
    be chained together to formulate multi-step plans while maximizing
    reward and prabability of successfully reaching the goal.
    """
    def __init__(self, num_features, brain):
        """
        Get the Model set up by allocating its variables.

        Parameters
        ----------
        brain : Brain
            The Brain to which this model belongs. Some of the brain's
            parameters are useful in initializing the model.
        num_features : int
            The total number of features allowed in this model.
        """
        # num_features : int
        #     The maximum number of features that the model can expect
        #     to incorporate. Knowing this allows the model to
        #     pre-allocate all the data structures it will need.
        #     Add 2 features/goals that are internal to the model,
        #     An "always on" and a "nothing else is on".
        self.num_features = num_features + 2

        # previous_feature_activities,
        # feature_activities : array of floats
        #     Features are characterized by their
        #     activity, that is, their level of activation at each time step.
        #     Activity can vary between zero and one.
        self.previous_feature_activities = np.zeros(self.num_features)
        self.feature_activities = np.zeros(self.num_features)

        # feature_goals,
        # previous_feature_goals,
        # feature_goal_votes : array of floats
        #     Goals can be set for features.
        #     They are temporary incentives, used for planning and
        #     goal selection. These can vary between zero and one.
        #     Votes are used to help choose a new goal each time step.
        self.feature_goal_activities = np.zeros(self.num_features)
        self.previous_feature_goals = np.zeros(self.num_features)
        self.feature_goal_votes = np.zeros(self.num_features)

        # FAIs : array of floats
        #     Feature activity increases.
        #     Of particular interest to us are **increases** in
        #     feature activities. These tend to occur at a
        #     specific point in time, so they are particularly useful
        #     in building meaningful temporal sequences.
        self.FAIs = np.zeros(self.num_features)

        # prefix_curiosities,
        # prefix_occurrences,
        # prefix_activities,
        # prefix_rewards : 2D array of floats
        # sequence_occurrences : 3D array of floats
        #     The properties associated with each sequence and prefix.
        #     If N is the number of features,
        #     the size of 2D arrays is N**2 and the shape of
        #     3D arrays is N**3. As a heads up, this can eat up
        #     memory as M gets large. They are indexed as follows:
        #         index 0 : feature_1 (past)
        #         index 1 : feature_goal
        #         index 2 : feature_2 (future)
        #     The prefix arrays can be 2D because they lack
        #     information about the resulting feature.
        _2D_size = (self.num_features, self.num_features)
        #_3D_size = (self.num_features, self.num_features, self.num_features)
        # Making believe that everything has occurred once in the past
        # makes it easy to believe that it might happen again in the future.
        self.prefix_activities = np.zeros(_2D_size)
        self.prefix_credit = np.zeros(_2D_size)
        self.prefix_occurrences = np.ones(_2D_size)
        self.prefix_curiosities = np.zeros(_2D_size)
        self.prefix_rewards = np.zeros(_2D_size)
        #self.sequence_occurrences = np.ones(_3D_size)

        # prefix_decay_rate : float
        #     The rate at which prefix activity decays between time steps
        #     for the purpose of calculating reward and finding the outcome.
        #     Decay takes five times longer for each additional level.
        self.prefix_decay_rate = .5
        # credit_decay_rate : float
        #     The rate at which the trace, a prefix's credit for the
        #     future reward, decays with each time step.
        self.credit_decay_rate = .2#.25#.35 # 5

        # reward_update_rate : float
        #     The rate at which a prefix modifies its reward estimate
        #     based on new observations.
        self.reward_update_rate = 3e-2
        # curiosity_update_rate : float
        #     One of the factors that determines he rate at which
        #     a prefix increases its curiosity.
        self.curiosity_update_rate = 3e-2

        viz.set_up_visualization(self, brain)


    def step(self, feature_activities, brain_live_features, reward):
        """
        Update the model and choose a new goal.

        Parameters
        ----------
        feature_activities : array of floats
            The current activity levels of each of the features.
        live_features : array of floats
            A binary array of all features that have every been active.
        reward : float
            The reward reported by the world during the most recent time step.
        """
        live_features = self._update_activities(
            feature_activities, brain_live_features)

        # Update sequences before prefixes.
        #nb.update_sequences(
        #    live_features,
        #    self.FAIs,
        #    self.prefix_activities,
        #    self.sequence_occurrences)

        nb.update_prefixes(
            live_features,
            self.prefix_decay_rate,
            self.previous_feature_activities,
            self.feature_goal_activities,
            self.prefix_activities,
            self.prefix_occurrences)

        nb.update_rewards(
            live_features,
            self.reward_update_rate,
            reward,
            self.prefix_credit,
            self.prefix_rewards)

        nb.update_curiosities(
            live_features,
            self.curiosity_update_rate,
            self.prefix_occurrences,
            self.prefix_curiosities,
            self.previous_feature_activities,
            self.feature_activities,
            self.feature_goal_activities)

        self.feature_goal_votes = nb.calculate_goal_votes(
            self.num_features,
            live_features,
            self.prefix_rewards,
            self.prefix_curiosities,
            self.prefix_occurrences,
            #self.sequence_occurrences,
            self.feature_activities,
            self.feature_goal_activities)

        goal_index, max_vote = self._choose_feature_goals()

        nb.update_reward_credit(
            live_features,
            goal_index,
            max_vote,
            self.feature_activities,
            self.credit_decay_rate,
            self.prefix_credit)

        return self.feature_goal_activities[2:]


    def _update_activities(self, feature_activities, brain_live_features):
        """
        Calculate the change in feature activities and goals.

        Parameters
        ----------
        brain_live_features : array of ints
            The set of indices of features that have had some activity
            in their lifetime.
        feature_activities : array of floats
            The current activities of each of the features.
        """

        # Augment the feature_activities with the two internal features,
        # the "always on" (index of 0) and
        # the "null" or "nothing else is on" (index of 1).
        self.previous_feature_activities = self.feature_activities
        self.feature_activities = np.concatenate((
            np.zeros(2), feature_activities))
        live_features = brain_live_features + 2
        live_features = list(live_features)
        live_features = [0, 1] + live_features
        live_features = np.array(live_features).astype('int32')

        # Track the increases in feature activities.
        self.FAIs = np.maximum(
            self.feature_activities - self.previous_feature_activities, 0.)
        # Assign the always on and the null feature.
        self.FAIs[0] = 1.
        total_activity = np.sum(self.FAIs[2:])
        inactivity = max(1. - total_activity, 0.)
        self.FAIs[1] = inactivity

        return live_features


    def _choose_feature_goals(self):
        """
        Using the feature_goal_votes, choose a goal.
        """
        # Choose one goal at each time step, the feature with
        # the largest vote.
        self.previous_feature_goals = self.feature_goal_activities
        self.feature_goal_activities = np.zeros(self.num_features)
        max_vote = np.max(self.feature_goal_votes)
        goal_index = 0
        matches = np.where(self.feature_goal_votes == max_vote)[0]
        # If there is a tie, randomly select between them.
        goal_index = matches[np.argmax(
            np.random.random_sample(matches.size))]
        self.feature_goal_activities[goal_index] = 1.

        return goal_index, max_vote


    def visualize(self, brain):
        """
        Make a picture of the model.

        Parameters
        ----------
        brain : Brain
            The brain that this model belongs to.
        """
        viz.visualize(self, brain)
