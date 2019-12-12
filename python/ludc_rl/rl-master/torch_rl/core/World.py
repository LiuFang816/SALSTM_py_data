
class World(object):
    '''
    A World defines a classical MDP i.e states, transitions
    '''
    def __init__(self):
        '''
        Initialize a new world
        dynamic_actions_space is set to True is the set of action changes depending on the state of the World
        '''
        self.dynamic_action_space=False
        pass

    def action_space(self):
        '''
        This function returns a description of the set of possible actions. If this set is dynamic, the function has to be called each time the state of the world changes.

        Return:
             The set of possible actions
        '''
        pass

    def step(self, action):
        '''
        Update the environment based on one action

        Args:
            - action: the action made by the agent
        '''
        raise NonImplementedError
    
    def reset(self,**parameters):
        '''
        Reset the environment by sampling an initial state

        Args:
             - parameters: can be used to give information about how the initial state has to be generated
        '''
        raise NonImplementedError

    def clone(self):
        '''
        Clone the environment in its current state
        '''
        raise NonImplementedError
