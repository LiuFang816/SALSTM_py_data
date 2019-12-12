import torch
from gym.spaces import Box
import numpy as np
import torch_rl
from torch_rl.core import Sensor
from .spaces import PytorchBox
from .spaces import CheapPytorchBox
from .spaces import MultipleSpaces

'''
Defines different generic sensors, particulary used to build sensors based on some sensors
'''


class PytorchSensor(Sensor):
    '''
    Build a sensor that produces pytorch tensors based on a sensor with a box output space.
    '''
    def __init__(self,sensor):
        space=sensor.sensor_space()
        self.sensor=sensor
        if isinstance(space,Box):
            self._sensor_space = PytorchBox(space.low,space.high)
        else:
            raise NotImplementedError('PytorchSensor only works with Box output spaces')

    def observe(self, env):
        observation=self.sensor.observe(env)
        return torch.Tensor(observation)

    def sensor_space(self):
        return self._sensor_space

class MultipleSensors(Sensor):
    '''
    This sensors build a tuple based on the observations of the list of provided sensors

    Args:
        - sensors: the list of sensors to aggregate
    '''
    def __init__(self,*sensors):
        self.sensors=sensors
        s=[]
        for ss in sensors:
            s.append(ss.sensor_space())

        self._sensor_space=MultipleSpaces(*(tuple(s)))

    def observe(self,env):
        o=[]
        for s in self.sensors:
            o.append(s.observe(env))
        return tuple(o)

    def sensor_space(self):
        return self._sensor_space

class ConcatenationSensors(Sensor):
    '''
    Build an observation as a concatenation of multiple tensors issued from different sensors

    Args:
        - sensors: the list of sensors to aggregate (must be sensor producing pytorch tensors)
   '''
    def __init__(self,*sensors):
        self.sensor=MultipleSensors(*sensors)
        dim=[]
        for i in sensors:
            dim.append(i.sensor_space().low.shape)

        final_dim=dim[0]
        for k in range(len(sensors)-1):
            kk=k+1
            fdim=(final_dim[0]+dim[kk][0],)
            for t in range(len(final_dim)-1):
                assert(final_dim[t+1]==dim[kk][t+1])
                fdim=fdim+(final_dim[t+1],)
            final_dim=fdim

        self._sensor_space=CheapPytorchBox(final_dim)

    def observe(self,env):
        return torch.cat(self.sensor.observe(env))

    def sensor_space(self):
        return self._sensor_space

class FlattenSensor(Sensor):
    '''
    Flatten a sensor observation to a simple vector

    Args:
        - sensor: the input sensor
   '''
    def __init__(self,sensor):
        self.sensor=sensor
        low=torch.Tensor(sensor.sensor_space().low)
        high=torch.Tensor(sensor.sensor_space().high)
        self.ne=low.nelement()
        self._sensor_space=PytorchBox(low.view(self.ne).numpy(),high.view(self.ne).numpy())

    def observe(self,env):
        return self.sensor.observe(env).view(self.ne)

    def sensor_space(self):
        return self._sensor_space