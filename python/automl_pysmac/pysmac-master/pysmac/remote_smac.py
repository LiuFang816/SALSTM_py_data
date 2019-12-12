from __future__ import print_function, division, absolute_import

import sys
import os
import traceback
import socket
import subprocess
import resource
from pkg_resources import resource_filename
from math import ceil
import errno

import logging
import multiprocessing

import time

import pynisher




SMAC_VERSION = "smac-v2.10.03-master-778"

try:
    str=unicode #Python 2 backward compatibility
except NameError:
    pass        #Python 3 case



# takes a name and a tuple defining one parameter, registers that with the parser
# and returns the corresponding string for the SMAC pcs file and the type of the
# variable for later casting
def process_single_parameter_definition(name, specification):
    """
    A helper function to process a single parameter definition for further communication with SMAC.
    """

    data_type_mapping = {'integer': int, 'real': float}

    assert isinstance(specification, tuple), "The specification \"{}\" for {} is not valid".format(specification,name)
    assert len(specification)>1, "The specification \"{}\" for {} is too short".format(specification,name)

    if specification[0] not in {'real', 'integer', 'ordinal', 'categorical'}:
        raise ValueError("Type {} for {} not understood".format(specification[0], name))

    string = '{} {}'.format(name, specification[0])

    # numerical values
    if specification[0] in {'real', 'integer'}:
        dtype = data_type_mapping[specification[0]]
        if len(specification[1])!= 2:
            raise ValueError("Range {} for {} not valid for numerical parameter".format(specification[1], name))
        if specification[1][0] >= specification[1][1]:
            raise ValueError("Interval {} not not understood.".format(specification[1]))
        if not (specification[1][0] <= specification[2] and specification[2] <= specification[1][1]):
            raise ValueError("Default value for {} has to be in the specified range".format(name))

        if specification[0] == 'integer':
            if (type(specification[1][0]) != int) or (type(specification[1][1]) != int) or (type(specification[2]) != int):
                raise ValueError("Bounds and default value of integer parameter {} have to be integer types!".format(name))

        string += " [{0[0]}, {0[1]}] [{1}]".format(specification[1], specification[2])

        if ((len(specification) == 4) and specification[3] == 'log'):
            if specification[1][0] <= 0:
                raise ValueError("Range for {} cannot contain non-positive numbers.".format(name))
            string += " log"

    # ordinal and categorical types
    if (specification[0] in {'ordinal', 'categorical'}):

        if specification[2] not in specification[1]:
            raise ValueError("Default value {} for {} is not valid.".format(specification[2], name))

        # make sure all elements are of the same type
        if (len(set(map(type, specification[1]))) > 1):
            raise ValueError("Not all values of {} are of the same type!".format(name))

        dtype = type(specification[1][0])
        string += " {"+",".join(map(str, specification[1])) + '}' + ('[{}]'.format(specification[2]))

    return string, dtype


def process_parameter_definitions(parameter_dict):
    """
    A helper function to process all parameter definitions conviniently with just one call.

    This function takes the parametr definitions from the user, converts
    them into lines for SMAC's PCS format, and also creates a dictionary
    later used in the comunication with the SMAC process.

    :param paramer_dict: The user defined parameter configuration space

    """
    pcs_strings = []
    parser_dict={}

    for k,v in list(parameter_dict.items()):
        line, dtype = process_single_parameter_definition(k,v)
        parser_dict[k] = dtype
        pcs_strings.append(line)

    return (pcs_strings, parser_dict)



class remote_smac(object):
    """
    The class responsible for the TCP/IP communication with a SMAC instance.
    """

    udp_timeout=5
    """
    The default value for a timeout for the socket
    """

    def __init__(self, scenario_fn, additional_options_fn, seed, class_path, memory_limit, parser_dict, java_executable):
        """
        Starts SMAC in IPC mode. SMAC will wait for udp messages to be sent.
        """
        self.__parser = parser_dict
        self.__subprocess = None
        self.__logger = multiprocessing.get_logger()

        # establish a socket
        self.__sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.__sock.settimeout(3)
        self.__sock.bind(('', 0))
        self.__sock.listen(1)

        self.__port = self.__sock.getsockname()[1]
        self.__logger.debug('picked port %i'%self.__port)

        # build the java command
        cmds  = java_executable.split()
        if memory_limit is not None:
            cmds += ["-Xmx%im"%memory_limit]
        cmds +=    ["-XX:ParallelGCThreads=4",
                "-cp",
                class_path,
                "ca.ubc.cs.beta.smac.executors.SMACExecutor",
                "--scenario-file", scenario_fn,
                "--tae", "IPC",
                "--ipc-mechanism", "TCP",
                "--ipc-remote-port", str(self.__port),
                "--seed", str(seed)
                ]

        with open(additional_options_fn, 'r') as fh:
            for line in fh:
                name, value = line.strip().split(' ')
                cmds += ['--%s'%name, '%s'%value]

        self.__logger.debug("SMAC command: %s"%(' '.join(cmds)))

        self.__logger.debug("Starting SMAC in ICP mode")

        # connect the output to the logger if the appropriate level has been set
        if self.__logger.level < logging.WARNING:
            self.__subprocess = subprocess.Popen(cmds, stdout =sys.stdout, stderr = sys.stderr)
        else:
            with open(os.devnull, "w") as fnull:
                self.__subprocess = subprocess.Popen(cmds, stdout = fnull, stderr = fnull)

    def __del__(self):
        """ Destructor makes sure that the SMAC process is terminated if necessary. """
        # shut the subprocess down on 'destruction'
        if not (self.__subprocess is None):
            self.__subprocess.poll()
            if self.__subprocess.returncode == None:
                self.__subprocess.kill()
                self.__logger.debug('SMAC had to be terminated')
            else:
                self.__logger.debug('SMAC terminated with returncode %i', self.__subprocess.returncode)


    def next_configuration(self):
        """ Method that queries the next configuration from SMAC.

        Connects to the socket, reads the message from SMAC, and
        converts into a proper Python representation (using the proper
        types). It also checks whether the SMAC subprocess is still alive.

        :returns: either a dictionary with a configuration, or None if SMAC has terminated
        """
        
        self.__logger.debug('trying to retrieve the next configuration from SMAC')
        self.__sock.settimeout(self.udp_timeout)
        self.__conn, addr = self.__sock.accept()
        while True:
            try:

                fconn = self.__conn.makefile('r')
                config_str = fconn.readline()
                break
            except socket.timeout:
                # if smac already terminated, there is nothing else to do
                if self.__subprocess.poll() is not None:
                    self.__logger.debug("SMAC subprocess is no longer alive!")
                    return None
                #otherwise there is funny buisiness going on!
                else:
                    self.__logger.debug("SMAC has not responded yet, but is still alive. Will keep waiting!")
                    continue
            except socket.error as e:
            #    continue
                if e.args[0] == errno.EAGAIN:
                    self.__logger.debug("Socket to SMAC process was empty, will continue to wait.")
                    time.sleep(1)
                    continue
                else:
                    raise
            except:
                raise

        self.__logger.debug("SMAC message: %s"%config_str)

        los = config_str.replace('\'','').split() # name is shorthand for 'list of strings'
        config_dict={}

        config_dict['instance']      = int(los[0][3:])
        config_dict['instance_info'] = str(los[1])
        config_dict['cutoff_time']   = float(los[2])
        config_dict['cutoff_length'] = float(los[3])
        config_dict['seed']          = int(los[4])


        for i in range(5, len(los), 2):
            config_dict[ los[i][1:] ] = self.__parser[ los[i][1:] ]( los[i+1])

        self.__logger.debug("Our interpretation: %s"%config_dict)
        return (config_dict)

    def report_result(self, result_dict):
        """Method to report the latest run results back to SMAC.

        This method communicates the results from the last run back to SMAC.

        :param result_dict: dictionary with the keys 'value', 'status', and 'runtime'.
        :type result_dic: dict
        """

        # for propper printing, we have to convert the status into unicode
        result_dict['status'] = result_dict['status'].decode()
        s = 'Result for SMAC: {0[status]}, {0[runtime]}, 0, {0[value]}, 0\
            '.format(result_dict)
        self.__logger.debug(s)
        self.__conn.sendall(s.encode())
        self.__conn.close();



def remote_smac_function(only_arg):
    """
    The function that every worker from the multiprocessing pool calls
    to perform a separate SMAC run.

    This function is not part of the API that users should access, but
    rather part of the internals of pysmac. Due to the limitations of the
    multiprocessing module, it can only take one argument which is a
    list containing important arguments in a very specific order. Check
    the source code if you want to learn more.

    """
    try:
        scenario_file, additional_options_fn, seed, function, parser_dict,\
          memory_limit_smac_mb, class_path, num_instances, mem_limit_function,\
          t_limit_function, deterministic, java_executable, timeout_quality = only_arg

        logger = multiprocessing.get_logger()

        smac = remote_smac(scenario_file, additional_options_fn, seed,
                               class_path, memory_limit_smac_mb,parser_dict, java_executable)

        logger.debug('Started SMAC subprocess')

        num_iterations = 0

        while True:
            config_dict = smac.next_configuration()

            # method next_configuration checks whether smac is still alive
            # if it is None, it means that SMAC has finished (for whatever reason)
            if config_dict is None:
                break

            # delete the unused variables from the dict
            if num_instances is None:
                del config_dict['instance']

            del config_dict['instance_info']
            del config_dict['cutoff_length']
            if deterministic:
                del config_dict['seed']

            current_t_limit = int(ceil(config_dict.pop('cutoff_time')))
            # only restrict the runtime if an initial cutoff was defined
            current_t_limit = None if t_limit_function is None else current_t_limit
            current_wall_time_limit =  None if current_t_limit is None else 10*current_t_limit

            # execute the function and measure the time it takes to evaluate
            wrapped_function = pynisher.enforce_limits(
                mem_in_mb=mem_limit_function,
                cpu_time_in_s=current_t_limit,
                wall_time_in_s=current_wall_time_limit,
                grace_period_in_s = 1)(function)

            # workaround for the 'Resource temporarily not available' error on
            # the BaWue cluster if to many processes were spawned in a short
            # period. It now waits a second and tries again for 8 times.
            num_try = 1
            while num_try <= 8:
                try:
                    start = time.time()
                    res = wrapped_function(**config_dict)
                    wall_time = time.time()-start
                    cpu_time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime
                    break
                except OSError as e:
                    if e.errno == 11:
                        logger.warning('Resource temporarily not available. Trail {} of 8'.format(num_try))
                        time.sleep(1)
                    else:
                        raise
                except:
                    raise
                finally:
                    num_try += 1
            if num_try == 9:
                logger.warning('Configuration {} crashed 8 times, giving up on it.'.format(config_dict))
                res = None

            if res is not None:
                try:
                    logger.debug('iteration %i:function value %s, computed in %s seconds'%(num_iterations, str(res), str(res['runtime'])))
                except (TypeError, AttributeError, KeyError, IndexError):
                    logger.debug('iteration %i:function value %s, computed in %s seconds'%(num_iterations, str(res),cpu_time))
                except:
                    raise
            else:
                logger.debug('iteration %i: did not return in time, so it probably timed out'%(num_iterations))


            # try to infere the status of the function call:
            # if res['status'] exsists, it will be used in 'report_result'
            # if there was no return value, it has either crashed or timed out
            # for simple function, we just use 'SAT'

            result_dict = {
                        'value' : timeout_quality,
                        'status': b'CRASHED' if res is None else b'SAT',
                        'runtime': cpu_time
                        }

            if res is not None:
                if isinstance(res, dict):
                    result_dict.update(res)
                else:
                    result_dict['value'] = res

            # account for timeeouts
            if not current_t_limit is None:
                if ( (result_dict['runtime'] > current_t_limit-2e-2) or
                        (wall_time >= 10*current_t_limit) ):
                    result_dict['status']=b'TIMEOUT'

            # set returned quality to default in case of a timeout
            if result_dict['status'] == b'TIMEOUT':
                result_dict['value'] = result_dict['value'] if timeout_quality is None else timeout_quality

            smac.report_result(result_dict)
            num_iterations += 1
    except:
        traceback.print_exc() # to see the traceback of subprocesses
