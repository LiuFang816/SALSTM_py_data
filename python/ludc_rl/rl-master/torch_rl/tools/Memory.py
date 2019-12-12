import math
import torch

class Memory(object):
    '''
    Implements a memory of scalar values. The memory can have an unlimited size, or a fixed size. When having a fixed size, the memory is a circular memory (the oldest value s removed from the memory when a new value is pushed)

    Args:
            - size_memory: if None, the memory as an infinte size.
    '''
    def __init__(self,size_memory=None):
        '''
        Initialize a memory

        Args:
            - size_memory: if None, the memory as an infinte size.
        '''
        self.size_memory=size_memory
        if (self.size_memory is None):
            self.memory=torch.Tensor(1000)
        else:
            self.memory=torch.Tensor(self.size_memory)
        self.position_in_memory=0
        self.loop=False

    def push(self,x):
        '''
        Add a value to the memory

        Args:
            - x: the value
        '''
        self.memory[self.position_in_memory]=x
        self.position_in_memory=self.position_in_memory+1

        if (self.size_memory is None):
            if (self.position_in_memory==self.memory.size(0)):
                self.memory.resize_(self.memory.size(0)+1000)
        else:
            if (self.position_in_memory==self.memory.size(0)):
                if (self.loop==False):
                    self.loop=True
                self.position_in_memory=0

    def size(self):
        '''
        Return:
            - the number of values in the memory
        '''
        if (self.size_memory is None):
            return self.position_in_memory
        else:
            if (not self.loop):
                return self.position_in_memory
            else:
                return self.memory.size(0)

    def random(self):
        '''
        Return:
            - Returns a random value stored in the memory
        '''
        idx=math.randint(0,self.size()-1)
        return(self.memory[idx])

    def mean(self):
        '''
        Return:
            - Returns the mean of the values stored in the memory
        '''
        if ((self.size_memory is None) or (self.loop==False)):
            return self.memory.narrow(0,0,self.position_in_memory).mean()
        else:
            return self.memory.mean()

    def get_memory(self):
        '''
        Return:
            - Return a tensor of values stored in the memory, order by increasing freshness.
        '''
        if ((self.size_memory is None) or (self.loop == False)):
            return self.memory.narrow(0, 0, self.position_in_memory)
        else:
            return torch.cat([self.memory.narrow(0,self.position_in_memory,self.memory.size(0)-self.position_in_memory),self.memory.narrow(0,0,self.position_in_memory)])



