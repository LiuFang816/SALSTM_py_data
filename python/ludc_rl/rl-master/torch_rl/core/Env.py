import gym
from gym.utils import seeding


class Env(gym.Env):
    '''
    A reward-based oepnAI Gym environment built based on a (world,reward,task) triplet
    '''
    def __init__(self,world,reward,sensor):
        self.world=world
        self.reward=reward
        self.sensor=sensor
        self._seed()
        self.action_space=self.world.action_space()
        self.observation_space=self.sensor.sensor_space()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        self.world.step(action)
        immediate_reward=self.reward.reward(self.world)
        observation=self.sensor.observe(self.world)
        finished=self.reward.finished(self.world)
        return observation,immediate_reward,finished,None

    def _reset(self):
        self.world.reset()
        return self.sensor.observe(self.world)

    def _render(self, mode='human', close=False):
            raise NotImplementedError


