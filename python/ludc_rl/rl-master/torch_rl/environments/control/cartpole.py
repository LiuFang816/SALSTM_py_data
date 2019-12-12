import torch_rl.core
from gym.spaces import Discrete
from gym.spaces import Box
import numpy as np
import math
import torch_rl.core.sensors as sensors

class Sensor(torch_rl.core.Sensor):
    """ Returns the state of the cartpole as a (1,4) tensor"""

    def __init__(self):
        m = np.array([1000, 1000, 1000, 1000])
        self._sensor_space = Box(-m, m)

    # Returns an observation over a particular environment.
    def observe(self, env):
        return env.state

    # A description of the space of the data returned by the sensor
    def sensor_space(self):
        return self._sensor_space


class World(torch_rl.core.World):
    """ The cartpole physical world"""

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02
        self._action_space = Discrete(2)
        self.display = None
        self.viewer = None

    def action_space(self):
        return self._action_space

    def step(self, action):
        assert (action == 0 or action == 1), "CartPole: problem with action"
        x = self.state[0]
        x_dot = self.state[1]
        theta = self.state[2]
        theta_dot = self.state[3]

        force = 0
        if (action == 1):
            force = self.force_mag
        else:
            force = -self.force_mag

        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
        self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state[0] = x
        self.state[1] = x_dot
        self.state[2] = theta
        self.state[3] = theta_dot

    def reset(self, **parameters):
        a = np.arange(4)
        a.fill(-0.05)
        self.state = np.random.rand(4) * 0.1 + a

class Task(torch_rl.core.Reward):
    """ This task is the RL task defined in openAI Gym. """
    def __init__(self):
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

    def reward(self,world):
        return 1

    def finished(self,world):
        done = ( world.state[0] < -self.x_threshold
                or world.state[0] > self.x_threshold
                or world.state[2] < -self.theta_threshold_radians
                or world.state[2] > self.theta_threshold_radians)
        return done
