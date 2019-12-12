# ==================================================================================
# Copyright (c) 2016, Brain Corporation
#
# This software is released under Creative Commons
# Attribution-NonCommercial-ShareAlike 3.0 (BY-NC-SA) license.
# Full text available here in LICENSE.TXT file as well as:
# https://creativecommons.org/licenses/by-nc-sa/3.0/us/legalcode
#
# In summary - you are free to:
#
#    Share - copy and redistribute the material in any medium or format
#    Adapt - remix, transform, and build upon the material
#
# The licensor cannot revoke these freedoms as long as you follow the license terms.
#
# Under the following terms:
#    * Attribution - You must give appropriate credit, provide a link to the
#                    license, and indicate if changes were made. You may do so
#                    in any reasonable manner, but not in any way that suggests
#                    the licensor endorses you or your use.
#    * NonCommercial - You may not use the material for commercial purposes.
#    * ShareAlike - If you remix, transform, or build upon the material, you
#                   must distribute your contributions under the same license
#                   as the original.
#    * No additional restrictions - You may not apply legal terms or technological
#                                   measures that legally restrict others from
#                                   doing anything the license permits.
# ==================================================================================

import abc


class ExecutionManager(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def start(self):
        """
        This will be called right before the simulation starts
        """
        pass

    @abc.abstractmethod
    def fast_action(self):
        """
        This is the time between steps of execution
        Data is consistent, but keep this piece absolutely minimal
        """
        pass

    @abc.abstractmethod
    def slow_action(self):
        """
        This is while the workers are running. You may do a lot of work here
        (preferably not more than the time of execution of workers).
        """
        pass

    @abc.abstractmethod
    def running(self):
        """
        While returning True the simulation will keep going.
        """

    @abc.abstractmethod
    def finish(self):
        """
        Called at the end of execution.
        """


class AbstractSignalProvider(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def start(self):
        """
        This will be called right before the simulation starts
        """
        pass

    @abc.abstractmethod
    def get_signal(self, name, time):
        """
        Get the signal for the current time step if time is zero, or relative if time is non zero
        """
        pass

    @abc.abstractmethod
    def advance(self):
        """
        Move to the next timestep
        """
        pass

    @abc.abstractmethod
    def finish(self):
        """
        Called at the end of execution.
        """
