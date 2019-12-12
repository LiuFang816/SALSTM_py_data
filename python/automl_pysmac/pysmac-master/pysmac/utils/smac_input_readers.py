import re

def read_pcs(filename):
    ''' Function to read a SMAC pcs file (format according to version 2.08).

    :param filename: name of the pcs file to be read
    :type filename: str
    :returns: tuple -- (parameters as a dict, conditionals as a list, forbiddens as a list)
    '''

    num_regex = "[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?"
    FLOAT_REGEX = re.compile("^[ ]*(?P<name>[^ ]+)[ ]*\[(?P<range_start>%s)[ ]*,[ ]*(?P<range_end>%s)\][ ]*\[(?P<default>[^#]*)\](?P<misc>.*)$" %(num_regex,num_regex))
    CAT_REGEX = re.compile("^[ ]*(?P<name>[^ ]+)[ ]*{(?P<values>.+)}[ ]*\[(?P<default>[^#]*)\](?P<misc>.*)$")
    COND_REGEX = re.compile("^[ ]*(?P<cond>[^ ]+)[ ]*\|[ ]*(?P<head>[^ ]+)[ ]*in[ ]*{(?P<values>.+)}(?P<misc>.*)$")
    FORBIDDEN_REGEX = re.compile("^[ ]*{(?P<values>.+)}(?P<misc>.*)$")

    param_dict = {} # name -> ([begin, end], default, flags)
    conditions = []
    forbiddens = []

    with open(filename) as fp:
        for line in fp:
            #remove line break and white spaces
            line = line.strip("\n").strip(" ")

            #remove comments
            if line.find("#") > -1:
                line = line[:line.find("#")] # remove comments

            # skip empty lines
            if line  == "":
                continue

            # categorial parameter
            cat_match = CAT_REGEX.match(line)
            if cat_match:
                name = cat_match.group("name")
                values = set([x.strip(" ") for x in cat_match.group("values").split(",")])
                default = cat_match.group("default")

                #logging.debug("CATEGORIAL: %s %s {%s} (%s)" %(name, default, ",".join(map(str, values)), type_))
                param_dict[name] = (values, default)

            float_match = FLOAT_REGEX.match(line)
            if float_match:
                name = float_match.group("name")
                values = [float(float_match.group("range_start")), float(float_match.group("range_end"))]
                default = float(float_match.group("default"))

                #logging.debug("PARAMETER: %s %f [%s] (%s)" %(name, default, ",".join(map(str, values)), type_))
                param_dict[name] = (values, default)
                if "i" in float_match.group("misc"):
                    param_dict[name] += ('int',)
                if "l" in float_match.group("misc"):
                    param_dict[name] += ('log',)

            cond_match = COND_REGEX.match(line)
            if cond_match:
                #logging.debug("CONDITIONAL: %s | %s in {%s}" %(cond, head, ",".join(values)))
                conditions.append(line)

            forbidden_match = FORBIDDEN_REGEX.match(line)
            if forbidden_match:
                #logging.debug("FORBIDDEN: {%s}" %(values))
                forbiddens.append(line)

    return param_dict, conditions, forbiddens

def write_pcs(pcs_filename, parameters, forbiddens, conditionals):
    ''' Function to write a SMAC PCS file'''
    with open(pcs_filename, 'w') as out:
        # Parameters
        out.write("# Parameters\n")
        for param, info in parameters.iteritems():
            if param == 'algorithm': # Handle this specially because merge_configuration_spaces doesn't return a set
                values = set(info[1])
                default = info[2]
            else:
                values = info[0]
                default = info[1]
            if isinstance(values, set): # Categorical
                line = "%(param)s {%(values)s} [%(default)s]" % dict(param=param, values=",".join(values), default=default)
            else:
                _type = '' if len(info) == 2 else info[2][0]
                if _type == 'i':
                    values = map(int, info[0])
                    default = int(info[1])
                line = "%(param)s %(values)s [%(default)s] %(_type)s" % dict(param=param, values=values, default=default, _type=_type)
            out.write(line +'\n')

        # Conditionals
        out.write("# Conditionals\n")
        for conditional in conditionals:
            out.write(conditional +'\n')

        out.write("# Forbidden\n")
        # Forbidden
        for forbidden in forbiddens:
            out.write(forbidden +'\n')

def read_scenario_file(fn):
    """ Small helper function to read a SMAC scenario file.

    :returns : dict -- (key, value) pairs are (variable name, variable value)
    """

    # translate the difference option names to a canonical name
    scenario_option_names = {'algo-exec' : 'algo',
                        'algoExec': 'algo',
                        'algo-exec-dir': 'execdir',
                        'exec-dir' : 'execdir',
                        'execDir' : 'execdir',
                        'algo-deterministic' : 'deterministic',
                        'paramFile' : 'paramfile',
                        'pcs-file' :'paramfile',
                        'param-file' : 'paramfile',
                        'run-obj' : 'run_obj',
                        'run-objective' : 'run_obj',
                        'runObj' : 'run_obj',
                        'overall_obj' : 'overall_obj',
                        'intra-obj' : 'overall_obj',
                        'intra-instance-obj'  : 'overall_obj',
                        'overall-obj'  : 'overall_obj',
                        'intraInstanceObj' : 'overall_obj',
                        'overallObj' : 'overall_obj',
                        'intra_instance_obj' : 'overall_obj',
                        'algo-cutoff-time' : 'cutoff_time',
                        'target-run-cputime-limit' : 'cutoff_time',
                        'target_run_cputime_limit'  : 'cutoff_time',
                        'cutoff-time' : 'cutoff_time',
                        'cutoffTime' : 'cutoff_time',
                        'cputime-limit' : 'tunerTimeout',
                        'cputime_limit' : 'tunerTimeout',
                        'tunertime-limit' : 'tunerTimeout',
                        'tuner-timeout' : 'tunerTimeout',
                        'tunerTimeout' : 'tunerTimeout',
                        'wallclock_limit' : 'wallclock-limit',
                        'runtime-limit' : 'wallclock-limit',
                        'runtimeLimit' : 'wallclock-limit',
                        'wallClockLimit'  : 'wallclock-limit',
                        'output-dir' : 'outdir',
                        'outputDirectory' : 'outdir',
                        'instances' : 'instance_file',
                        'instance-file' : 'instance_file' ,
                        'instance-dir' : 'instance_file',
                        'instanceFile' : 'instance_file',
                        'i' : 'instance_file',
                        'instance_seed_file' : 'instance_file',
                        'test-instances' : 'test_instance_file',
                        'test-instance-file'  : 'test_instance_file',
                        'test-instance-dir' : 'test_instance_file',
                        'testInstanceFile' : 'test_instance_file',
                        'test_instance_file' : 'test_instance_file',
                        'test_instance_seed_file' : 'test_instance_file',
                        'feature-file' : 'feature_file',
                        'instanceFeatureFile' : 'feature_file',
                        'feature_file' : 'feature_file'
                    }

    scenario_dict = {}
    with open(fn, 'r') as fh:
        for line in fh.readlines():
            #remove comments
            if line.find("#") > -1:
                line = line[:line.find("#")]

            # skip empty lines
            if line  == "":
                continue
            if "=" in line:
                tmp = line.split("=")
                tmp = [' '.join(s.split()) for s in tmp]
            else:
                tmp = line.split()
            scenario_dict[scenario_option_names.get(tmp[0],tmp[0])] = " ".join(tmp[1:])
    return(scenario_dict)
