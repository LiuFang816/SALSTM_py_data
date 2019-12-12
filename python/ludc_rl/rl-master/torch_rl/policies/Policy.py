class Policy(object):
    '''
    A policy is a dsitribution P(a_t|o_1,...,o_t) where o_1,...,o_t is the sequence of previoulsy received observations
    '''
    def __init__(self,action_space):
        '''
        Initialize a policy over a particular action space

        Args:
            - action_space: the action space of the policy
        '''
        self.action_space=action_space

    def start_episode(self,**parameters):
        '''
        Tell the policy that a new episode is starting

        Args:
            - parameters:  aditionnal parameters can be provided
        '''
        pass

    def end_episode(self):
        '''
        Tell the policy that an episode just ends. A feedback can be provided to the policy at the end of the episode

        Args:
            -parameters: informations provided to the policy at the end of the episode
        '''
        pass

    def observe(self,observation):
        '''
        provides a new observation to the policy

        Args:
            - observation: the observation (usually coming from a sensor)
        '''
        pass

    def sample(self):
        '''
        Returns a sampled action based on the set of previoulsy received observations (in the episode)

        Returns:
            - an action
        '''
        pass

