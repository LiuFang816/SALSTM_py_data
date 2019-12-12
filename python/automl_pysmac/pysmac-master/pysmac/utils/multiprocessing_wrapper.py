#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import multiprocessing
import multiprocessing.pool

class NoDaemonProcess(multiprocessing.Process):
    """
    A subclass to avoid the subprocesses used for the individual SMAC runs to be deamons.
    
    As it turns out, the Java processes running SMAC cannot be deamons.
    To change the default behavior of the multiprocessing module, one has
    to derive a subclass and overwrite the _get_deamon, _set_deamon methods
    appropriately.
    """
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class MyPool(multiprocessing.pool.Pool):
    """Subclass to use the NoDeamonProcesses as workers in a Pool."""
    Process = NoDaemonProcess
