"""
Connect a world with a brain and set it to running.
"""
from __future__ import print_function
import numpy as np

from becca.brain import Brain

def run(world, restore=False):
    """
    Run Becca with a world.

    Connect the brain and the world together and run them for as long
    as the world dictates.

    Parameters
    ----------
    world : World
        The world that Becca will learn.
        See the world.py documentation for a full description.
    restore : bool, optional
        If restore is True, try to restore the brain
        from a previously saved
        version, picking up where it left off.
        Otherwise it create a new one. The default is False.

    Returns
    -------
    performance : float
        The performance of the brain over its lifespan, measured by the
        average reward it gathered per time step.
    """
    brain_name = '{0}_brain'.format(world.name)
    #if 'world.log_directory' in locals() and world.log_directory is not None:
    #if world.log_directory is not None:
    try:
        brain = Brain(world.num_sensors,
                      world.num_actions,
                      brain_name=brain_name,
                      log_directory=world.log_directory)
    except:
        brain = Brain(world.num_sensors,
                      world.num_actions,
                      brain_name=brain_name)

    if restore:
        brain = brain.restore()

    try:
        brain.visualize_interval = world.brain_visualize_interval
        print('Brain visualize interval set to',
              world.brain_visualize_interval)
    except:
        pass

    # Start at a resting state.
    actions = np.zeros(world.num_actions)
    sensors, reward = world.step(actions)

    # Repeat the loop through the duration of the existence of the world:
    # sense, act, repeat.
    while world.is_alive():
        actions = brain.sense_act_learn(sensors, reward)
        sensors, reward = world.step(actions)

        # Create visualizations.
        if brain.timestep % brain.visualize_interval == 0:
            brain.visualize(world)
        if world.timestep % world.visualize_interval == 0:
            world.visualize(brain)

    # Wrap up the run.
    try:
        world.close_world(brain)
    except AttributeError:
        print("Closing", world.name_long)

    performance = brain.report_performance()
    return performance
