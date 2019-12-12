"""
The base world on which all the other worlds are based.
"""
from __future__ import print_function
import numpy as np


class World(object):
    """
    The base class for creating a new world.
    """
    def __init__(self, lifespan=None):
        """
        Initialize a new world with some benign default values.

        Parameters
        ----------
        lifespan : int, optional
            The number of time steps that the world will be
            allowed to continue.
        """
        # lifespan : float
        #     The number of time steps for which the world should continue
        #     to exist.
        if lifespan is None:
            self.lifespan = 10 ** 5
        else:
            self.lifespan = lifespan
        # timestep : int
        #     The number of time steps that the world has already been through.
        #     Starting at -1 allows for an intialization pass.
        self.timestep = -1
        # world_visualization_period : int
        #     How often to turn the world into a picture.
        self.visualize_interval = 1e6
        # name : String
        #     The name of the world.
        self.name = 'abstract base world'
        # num_actions, num_sensors : int
        #     The number of actions and sensors, respectively.
        #     These will likely be overridden in any subclass
        self.num_sensors = 0
        self.num_actions = 0
        # sensors, actions : array of floats
        #     The arrays that represent the full set of sensed observations
        #     and intended actions, updated at each time step.
        self.sensors = np.zeros(self.num_sensors)
        self.actions = np.zeros(self.num_actions)
        # reward : float
        #     The feedback signal on the goodness of an brain's experience.
        #     0 is neutral.
        #     1 is very, very good.
        #     -1 is very, very bad.
        self.reward = 0


    def step(self, actions):
        """
        Take a time step through an empty world that does nothing.

        Parameters
        ----------
        actions : array of floats
            The set of actions that the world can be expected to execute.

        Returns
        -------
        sensors : array of floats
            The current values of each of those sensors in the world.
        reward : float
            The current reward provided by the world.
        """
        self.timestep += 1
        self.sensors = np.zeros(self.num_sensors)
        self.reward = 0
        return self.sensors, self.reward


    def is_alive(self):
        """
        Check whether the world is alive.

        Once more than lifespan time steps have been completed,
        stop running.

        Returns
        -------
        If False, the world has come to an end.
        """
        return self.timestep < self.lifespan


    def visualize(self, brain):
        """
        Show the user the state of the world.
        """
        print('{0} is {1} time steps old.'.format(self.name, self.timestep))
        print('The brain is {0} time steps old.'.format(brain.timestep))

