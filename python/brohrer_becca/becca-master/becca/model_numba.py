"""
Numba functions that support model.py
"""

from __future__ import print_function
from numba import jit
import numpy as np


@jit(nopython=True)
def update_sequences(
        live_features,
        new_FAIs,
        prefix_activities,
        sequence_occurrences):
    """
    Update the number of occurrences of each sequence.

    The new sequence activity, n, is
        n = p * f, where
    p is the prefix activities from the previous time step and
    f is the new_FAIs

    These are temporarily disabled, since they haven't proven themselves
    absolutely necessary yet.
    """
    small = .1
    for j_feature in live_features:
        if new_FAIs[j_feature] < small:
            continue
        for i_goal in live_features:
            for i_feature in live_features:
                if prefix_activities[i_feature][i_goal] < small:
                    continue
                sequence_occurrences[i_feature][i_goal][j_feature] += (
                    prefix_activities[i_feature][i_goal] *
                    new_FAIs[j_feature])
    return


@jit(nopython=True)
def update_prefixes(
        live_features,
        prefix_decay_rate,
        previous_feature_activities,
        feature_goal_activities,
        prefix_activities,
        prefix_occurrences):
    """
    Update the activities and occurrences of the prefixes.

    The new activity of a feature-goal prefix, n,  is
         n = f * g, where
    f is the previous_FAI and
    g is the current goal_increase.

    p, the prefix activity, is a decayed version of n.
    """
    for i_feature in live_features:
        for i_goal in live_features:
            prefix_activities[i_feature][i_goal] *= (
                1. - prefix_decay_rate)

            new_prefix_activity = (previous_feature_activities[i_feature] *
                                   feature_goal_activities[i_goal])
            prefix_activities[i_feature][i_goal] += new_prefix_activity
            prefix_activities[i_feature][i_goal] = min(
                prefix_activities[i_feature][i_goal], 1.)

            # Increment the lifetime sum of prefix activity.
            prefix_occurrences[i_feature][i_goal] += (
                prefix_activities[i_feature][i_goal])
    return


@jit(nopython=True)
def update_rewards(
        live_features,
        reward_update_rate,
        reward,
        prefix_credit,
        prefix_rewards):
    """
    Assign credit for the current reward to any recently active prefixes.

    Increment the expected reward associated with each prefix.
    The size of the increment is larger when:
        1. the discrepancy between the previously learned and
            observed reward values is larger and
        2. the prefix activity is greater.
    Another way to say this is:
    If either the reward discrepancy is very small
    or the sequence activity is very small, there is no change.
    """
    for i_feature in live_features:
        for i_goal in live_features:
            prefix_rewards[i_feature][i_goal] += (
                (reward - prefix_rewards[i_feature][i_goal]) *
                prefix_credit[i_feature][i_goal] *
                reward_update_rate)
    return


@jit(nopython=True)
def update_curiosities(
        live_features,
        curiosity_update_rate,
        prefix_occurrences,
        prefix_curiosities,
        previous_feature_activities,
        feature_activities,
        feature_goal_activities):
    """
    Use a collection of factors to increment the curiosity for each prefix.
    """
    for i_feature in live_features:
        for i_goal in live_features:

            # Fulfill curiosity on the previous time step's goals.
            curiosity_fulfillment = (previous_feature_activities[i_feature] *
                                     feature_goal_activities[i_goal])
            prefix_curiosities[i_feature][i_goal] -= curiosity_fulfillment
            prefix_curiosities[i_feature][i_goal] = max(
                prefix_curiosities[i_feature][i_goal], 0.)

            # Increment the curiosity based on several multiplicative
            # factors.
            #     curiosity_update_rate : a constant
            #     uncertainty : an estimate of how much is not yet
            #         known about this prefix. It is a function of
            #         the total past occurrences.
            #     feature_activities : The activity of the prefix's feature.
            #         Only increase the curiosity if the feature
            #         corresponding to the prefix is active.
            uncertainty = 1. / (1. + 3. * prefix_occurrences[i_feature][i_goal])
            prefix_curiosities[i_feature][i_goal] += (
                curiosity_update_rate *
                uncertainty *
                feature_activities[i_feature])
    return


@jit(nopython=True)
def calculate_goal_votes(
        num_features,
        live_features,
        prefix_rewards,
        prefix_curiosities,
        prefix_occurrences,
        #sequence_occurrences,
        feature_activities,
        feature_goal_activities):
    """
    Let each prefix cast a vote for its goal, based on its value.

    For each prefix and its corresponding sequences calculate
    the expected value, v, of the goal.

            v = a * (r + c + s), where
        a is the activity of the prefix's feature,
        r is the prefix's reward,
        c is the prefix's curiosity, and
        s is the overall expected value of the prefix's sequences.

            s = sum((o / p) * (g + t)) / sum(o / p), where
        o is the number of occurrences of the sequence,
        p is the number of occurrences of the prefix,
        g is the goal value of the sequence's terminal feature, and
        t is the top-down plan value of the sequence's terminal feature.

    For each goal, track the largest value that is calculated and
    treat it as a vote for that goal.
    """
    small = .1
    feature_goal_votes = np.zeros(num_features)
    for i_feature in live_features:
        for i_goal in live_features:
            if feature_activities[i_feature] < small:
                goal_vote = -2.

            else:
                # Hold out this code for now. As far as I know, it works.
                # I'll be able to tell with more certainty after additional
                # testing.
                '''
                # Add up the value of sequences.
                weighted_values = 1.
                total_weights = 1.
                for j_feature in live_features:
                    weight = (
                        sequence_occurrences[i_feature][i_goal][j_feature]*
                        prefix_occurrences[i_feature][i_goal])
                    weighted_values += (
                        weight * feature_goal_activities[j_feature])
                    total_weights += weight
                sequence_value = weighted_values / total_weights
                '''
                # Add up the other value components.
                goal_vote = feature_activities[i_feature] * (
                    prefix_rewards[i_feature][i_goal] +
                    prefix_curiosities[i_feature][i_goal])# +
                    #sequence_value)

            # Compile the maximum goal votes for action selection.
            if goal_vote > feature_goal_votes[i_goal]:
                feature_goal_votes[i_goal] = goal_vote

    return feature_goal_votes


@jit(nopython=True)
def update_reward_credit(
        live_features,
        i_new_goal,
        max_vote,
        feature_activities,
        credit_decay_rate,
        prefix_credit):
    """
    Update the credit due each prefix for upcoming reward.
    """
    # Age the prefix credit.
    for i_feature in live_features:
        for i_goal in live_features:
            # Exponential discounting
            prefix_credit[i_feature][i_goal] *= (1. - credit_decay_rate)

    if max_vote > 0.:
        # Update the prefix credit.
        if i_new_goal > -1:
            for i_feature in live_features:
                # Accumulation strategy:
                # add new credit to existing credit, with a max of 1.
                prefix_credit[i_feature][i_new_goal] += (
                    feature_activities[i_feature])
                prefix_credit[i_feature][i_new_goal] = min(
                    prefix_credit[i_feature][i_new_goal], 1.)
    return
