import json
import functools
import re
import operator

import numpy as np

from pysmac.remote_smac import process_parameter_definitions


def convert_param_dict_types(param_dict, pcs):
    _, parser_dict = process_parameter_definitions(pcs)
    for k in param_dict:
        param_dict[k] = parser_dict[k](param_dict[k])
    return(param_dict)



def json_parse(fileobj, decoder=json.JSONDecoder(), buffersize=2048):
    """ Small function to parse a file containing JSON objects separated by a new line. This format is used in the live-rundata-xx.json files produces by SMAC.
    
    taken from http://stackoverflow.com/questions/21708192/how-do-i-use-the-json-module-to-read-in-one-json-object-at-a-time/21709058#21709058
    """
    buffer = ''
    for chunk in iter(functools.partial(fileobj.read, buffersize), ''):
        buffer += chunk
        buffer = buffer.strip(' \n')
        while buffer:
            try:
                result, index = decoder.raw_decode(buffer)
                yield result
                buffer = buffer[index:]
            except ValueError:
                # Not enough data to decode, read more
                break


def read_runs_and_results_file(fn):
    """ Converting a runs_and_results file into a numpy array.
    
    Almost all entries in a runs_and_results file are numeric to begin with.
    Only the 14th column contains the status which is encoded as ints by SAT = 1,
    UNSAT = 0, TIMEOUT = -1, everything else = -2.
    \n
    +-------+----------------+
    | Value | Representation |
    +=======+================+
    |SAT    |        2       |
    +-------+----------------+
    |UNSAT  |        1       |
    +-------+----------------+
    |TIMEOUT|        0       |
    +-------+----------------+
    |Others |       -1       |
    +-------+----------------+
    
    
    :returns: numpy_array(dtype = double) -- the data
    """
    # to convert everything into floats, the run result needs to be mapped
    def map_run_result(res):
         if b'TIMEOUT' in res:  return(0)
         if b'UNSAT' in res:    return(1) # note UNSAT before SAT, b/c UNSAT contains SAT!
         if b'SAT' in res:      return(2)
         return(-1)    # covers ABORT, CRASHED, but that shouldn't happen
    
    return(np.loadtxt(fn, skiprows=1, delimiter=',',
        usecols = list(range(1,14))+[15], # skip empty 'algorithm run data' column
        converters={13:map_run_result}, ndmin=2))


def read_paramstrings_file(fn):
    """ Function to read a paramstring file.
    Every line in this file corresponds to a full configuration. Everything is
    stored as strings and without knowledge about the pcs, converting that into
    any other type would involve guessing, which we shall not do here.
    
    :param fn: the name of the paramstring file
    :type fn: str
    :returns: dict -- with key-value pairs 'parameter name'-'value as string'
    
    """
    param_dict_list = []
    with open(fn,'r') as fh:
        for line in fh.readlines():
            # remove run id and single quotes
            line = line[line.find(':')+1:].replace("'","")
            pairs = [s.strip().split("=") for s in line.split(',')]
            param_dict_list.append({k:v for [k, v] in pairs})
    return(param_dict_list)


def read_validationCallStrings_file(fn):
    """Reads a validationCallString file into a list of dictionaries.
    
    :returns: list of dicts -- each dictionary contains 'parameter name' and 'parameter value as string' key-value pairs
    """
    param_dict_list = []
    with open(fn,'r') as fh:
        for line in fh.readlines()[1:]: # skip header line
            config_string = line.split(",")[1].strip('"')
            config_string = config_string.split(' ')
            tmp_dict = {}
            for i in range(0,len(config_string),2):
                tmp_dict[config_string[i].lstrip('-')] = config_string[i+1].strip("'")
            param_dict_list.append(tmp_dict)
    return(param_dict_list)


def read_validationObjectiveMatrix_file(fn):
    """ reads the run data of a validation run performed by SMAC.
    
    For cases with instances, not necessarily every instance is used during the
    configuration phase to estimate a configuration's performance. If validation
    is enabled, SMAC reruns parameter settings (usually just the final incumbent)
    on the whole instance set/a designated test set. The data from those runs
    is stored in separate files. This function reads one of these files.
    
    :param fn: the name of the validationObjectiveMatrix file
    :type fn: str
    
    :returns: dict -- configuration ids as keys, list of performances on each instance as values.
    
    .. todo::
       testing of validation runs where more than the final incumbent is validated
    """
    values = {}
    
    with open(fn,'r') as fh:
        header = fh.readline().split(",")
        num_configs = len(header)-2
        re_string = '\w?,\w?'.join(['"id\_(\d*)"', '"(\d*)"']  + ['"([0-9.]*)"']*num_configs)
        for line in fh.readlines():
            match = (re.match(re_string, line))
            values[int(match.group(1))] = list(map(float,list(map(match.group, list(range(3,3+num_configs))))))
    return(values)


def read_trajectory_file(fn):
    """Reads a trajectory file and returns a list of dicts with all the information.
    
    Due to the way SMAC stores every parameter's value as a string, the configuration returned by this function also has every value stored as a string. All other values, like "Estimated Training Preformance" and so on are floats, though.
    
    :param fn: name of file to read
    :type fn: str
    
    :returns: list of dicts -- every dict contains the keys: "CPU Time Used","Estimated Training Performance","Wallclock Time","Incumbent ID","Automatic Configurator (CPU) Time","Configuration"
    """
    return_list = []
    
    with open(fn,'r') as fh:
        header = list(map(lambda s: s.strip('"'), fh.readline().split(",")))
        l_info = len(header)-1
        for line in fh.readlines():
            tmp = line.split(",")
            tmp_dict = {}
            for i in range(l_info):
                tmp_dict[header[i]] = float(tmp[i])
            tmp_dict['Configuration'] = {}
            for i in range(l_info, len(tmp)):
                name, value = tmp[i].strip().split("=")
                tmp_dict['Configuration'][name] = value.strip("'").strip('"')
            return_list.append(tmp_dict)
    return(return_list)

def read_instances_file(fn):
    """Reads the instance names from an instace file
    
    :param fn: name of file to read
    :type fn: str
    :returns: list -- each element is a list where the first element is the instance name followed by additional information for the specific instance.
    """
    with open(fn,'r') as fh:
        instance_names = fh.readlines()
    return([s.strip().split() for s in instance_names])


def read_instance_features_file(fn):
    """Function to read a instance_feature file.
    
    :returns: tuple -- first entry is a list of the feature names, second one is a dict with 'instance name' - 'numpy array containing the features' key-value pairs
    """
    instances = {}
    with open(fn,'r') as fh:
        lines = fh.readlines()
        for line in lines[1:]:
            tmp = line.strip().split(",")
            instances[tmp[0]] = np.array(tmp[1:],dtype=np.double)
    return(lines[0].split(",")[1:], instances)
