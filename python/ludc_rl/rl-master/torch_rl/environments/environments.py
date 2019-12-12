from torch_rl.core.Env import *
import gym.spaces

class MappedEnv(Env):
    '''
    remap observations using a mapping function observation -> new observation
    '''
    def __init__(self, env, mapping_function,observation_space=None):
        self.env = env
        self.action_space = env.action_space
        self.mapping_function=mapping_function
        self.observation_space = observation_space
        self.metadata = env.metadata

    def _seed(self, seed=None):
        return self.env._seed(seed=seed)


    def _step(self, action):
        ob,r,f,i=self.env._step(action)
        no=self.mapping_function(ob)
        return no,r,f,i

    def _reset(self):
        o=self.env._reset()
        no=self.mapping_function(o)
        return no

    def _render(self, mode='human', close=False):
        return self.env._render(mode=mode, close=close)

class RemapDiscreteEnv(Env):
    '''
    Allows one to remap discrete actions to another discrete actions

    - Args:
    * actions: the list of authorized actions
    '''
    def __init__(self, env,actions):
        self.env = env
        self.action_space = gym.spaces.Discrete(len(actions))
        self.actions=actions
        self.observation_space = env.observation_space
        self.metadata=env.metadata


    def _seed(self, seed=None):
        return self.env._seed(seed=seed)


    def _step(self, action):
        na=self.actions[action]
        return self.env._step(na)

    def _reset(self):
        return self.env._reset()


    def _render(self, mode='human', close=False):
        return self.env._render(mode=mode, close=close)

class InfiniteEnv(Env):
    '''
    An environment which never finishes. When the initial environment is finished, this one will returns a zero reward, and always the same (last) observation
    '''
    def __init__(self,env):
        self.env=env
        self.action_space=env.world.action_space
        self.observation_space=env.observation_space
        self.is_finished=False
        self.metadata = env.metadata

    def _seed(self, seed=None):
        return self.env._seed(seed=seed)

    def _step(self, action):
        if (not self.is_finished):
            self.observation,immediate_reward,self.is_finished,info=self.env._step(action)
            return self.observation,immediate_reward,False,info
        else:
            return self.observation,0,False,None

    def _reset(self):
        return self.env._reset()


    def _render(self, mode='human', close=False):
        return self.env._render(mode=mode,close=close)


