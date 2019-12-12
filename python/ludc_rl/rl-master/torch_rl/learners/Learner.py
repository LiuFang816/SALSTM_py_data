

class Learner:
    '''
    This class aims at learning a policy
    '''
    def reset(self,**parameters):
        '''
        Args:
            - parameters: parameters for resetting the learner

        Returns:
            - Return the initial starting policy
        '''
        pass

    def step(self,**parameters):
        '''
        Make one learning step
        '''
        pass

    def get_policy(self,**parameters):
        '''
        Return the policy learned by the learner

        Returns:
            - the policy that is currently being learned
        '''
        pass