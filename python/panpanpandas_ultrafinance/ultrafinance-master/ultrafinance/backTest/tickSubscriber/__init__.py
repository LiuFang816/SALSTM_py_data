'''
Created on Nov 6, 2011

@author: ppa
'''
import abc
import uuid
import threading

class TickSubsriber(object):
    ''' tick subscriber '''
    __metaclass__ = abc.ABCMeta

    def __init__(self, name):
        ''' constructor '''
        self.__id = self.__generateId()
        self.__name = name
        self.__threadLock = threading.Lock()

    def __generateId(self):
        ''' generate id '''
        return uuid.uuid4()

    def __getId(self):
        ''' get id '''
        return self.__id

    def __getName(self):
        ''' get name '''
        return self.__name

    def preConsume(self, ticks):
        ''' override function '''
        pass

    @abc.abstractmethod
    def tickUpdate(self, ticks):
        ''' consume ticks '''
        return

    def orderExecuted(self, orderDict):
        ''' call back for executed order with order id, should be overridden '''
        return

    def complete(self):
        ''' complete operation '''
        pass

    @abc.abstractmethod
    def subRules(self):
        ''' call back from framework
            return (symbolRe, rules)
        '''
        return

    subId = property(__getId)
    name = property(__getName)
