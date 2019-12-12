from torch_rl.core.Task import Task

class Reward(Task):
    ''' Corresponds to a task where one wants to maximize a reward. The class correspnods to the immediate reward sent by the environment'''
    def __init__(self):
        pass

    def reward(self,world):
        raise NonImplementedError

