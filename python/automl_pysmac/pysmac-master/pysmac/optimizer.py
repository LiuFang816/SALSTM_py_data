from __future__ import print_function, division, absolute_import

import sys
import tempfile
import os
import shutil
import errno
import operator
import multiprocessing
import logging
import csv

from .utils.smac_output_readers import read_trajectory_file
import pysmac.remote_smac
from .utils.multiprocessing_wrapper import MyPool
from pysmac.utils.java_helper import check_java_version, smac_classpath


class SMAC_optimizer(object):
    """
    The main class of pysmac instanciated by the user.
    
    This is the class a user instanciates to use SMAC. Constructing the
    object does not start the minimization immediately. The user has to
    call the method minimize for the actual optimization. This design
    choice enables easy enough usage for novice users, but allows experts
    to change many of SMAC's parameters by editing the 'smac_options' dict
    """


    smac_options = {}
    """ A dict associated with the optimizer object that controlls options
    mainly for SMAC
    """



    # collects smac specific data that go into the scenario file
    def __init__(self, t_limit_total_s=None, mem_limit_smac_mb=None, working_directory = None, persistent_files=False, debug = False):
        """
        
        :param t_limit_total_s: the total time budget (in seconds) for the optimization. None means that no wall clock time constraint is enforced.
        :type t_limit_total_s: float
        :param mem_limit_smac_mb: memory limit for the Java Runtime Environment in which SMAC will be executed. None means system default.
        :type mem_limit_smac_mb: int
        :param working_directory: directory where SMACs output files are stored. None means a temporary directory will be created via the tempfile module.
        :type working_directory: str
        :param persistent_files: whether or note these files persist beyond the runtime of the optimization.
        :type persistent_files: bool
        :param debug: set this to true for debug information (pysmac and SMAC itself) logged to standard-out. 
        :type debug: bool
        """
        
        self.__logger = multiprocessing.log_to_stderr()
        if debug:
            self.__logger.setLevel(debug)
        else:
            self.__logger.setLevel(logging.WARNING)
        
        self.__t_limit_total_s = 0 if t_limit_total_s is None else int(t_limit_total_s)
        self.__mem_limit_smac_mb = None if (mem_limit_smac_mb is None) else int(mem_limit_smac_mb)
            
        self.__persistent_files = persistent_files
        
        # some basic consistency checks

        if (self.__t_limit_total_s < 0):
            raise ValueError('The total time limit cannot be nagative!')
        if (( self.__mem_limit_smac_mb is not None) and (self.__mem_limit_smac_mb <= 0)):
            raise ValueError('SMAC\'s memory limit has to be either None (no limit) or positive!')

        
        # create a temporary directory if none is specified
        if working_directory is None:
            self.working_directory = tempfile.mkdtemp()
        else:
            self.working_directory = working_directory
        
        self.__logger.debug('Writing output into: %s'%self.working_directory)
        
        # make some subdirs for output and smac internals
        self.__exec_dir = os.path.join(self.working_directory, 'exec')
        self.__out_dir  = os.path.join(self.working_directory, 'out' )

        for directory in [self.working_directory, self.__exec_dir, self.__out_dir]:
            try:
                os.makedirs(directory)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise
        
                
        # Set some of smac options
        # Most fields contain the standard values (as of SMAC 2.08.00).
        # All options from the smac manual can be accessed by
        # adding an entry to the dictionary with the appropriate name.
        # Some options will however have, at best, no effect, setting
        # others may even brake the communication.
        self.smac_options = {
            'algo-exec': 'echo 0',
            'run-obj': 'QUALITY',
            'validation': False,
            'cutoff_time': 3600,
            'intensification-percentage': 0.5,
            'numPCA': 7,
            'rf-full-tree-bootstrap': False,
            'rf-ignore-conditionality':False,
            'rf-num-trees': 10,
            'skip-features': True,
            'pcs-file': os.path.join(self.working_directory,'parameters.pcs'),
            'instances': os.path.join(self.working_directory ,'instances.dat'),
            'algo-exec-dir': self.working_directory,
            'output-dir': self.__out_dir,
            'console-log-level': 'OFF',
            'abort-on-first-run-crash': False,
            'overall_obj': 'MEAN',
            'scenario_fn': 'scenario.dat', # NOT a SMAC OPTION, but allows to
                                          # change the standard name (used
                                          # in Spysmac)
            'java_executable': 'java',    # NOT a SMAC OPTION; allows to 
                                          # specify a different java
                                          # binary and can be abused to 
                                          # pass additional arguments to it
            'timeout_quality':2.**127,    # not a SMAC option either
                                          # custamize the quality reported
                                          # to SMAC in case of a timeout
            }
        if debug:
            self.smac_options['console-log-level']='INFO'

    def __del__(self):
        """
        Destructor cleaning up after SMAC finishes depending on the persistent_files flag.
        """
        if not self.__persistent_files:
            shutil.rmtree(self.working_directory)

    def minimize(self, func, max_evaluations, parameter_dict, 
            conditional_clauses = [], forbidden_clauses=[],
            deterministic = True,
            num_train_instances = None, num_test_instances = None,
            train_instance_features = None,
            num_runs = 1, num_procs = 1, seed = 0,
            mem_limit_function_mb=None, t_limit_function_s= None):
        """
        Function invoked to perform the actual minimization given all necessary information.
        
        :param func: the function to be called
        :type func: callable
        :param max_evaluations: number of function calls allowed during the optimization (does not include optional validation).
        :type max_evaluations: int
        :param parameter_dict: parameter configuration space definition, see :doc:`pcs`.
        :type parameter_dict: dict
        :param conditional_clauses: list of conditional dependencies between parameters,  see :doc:`pcs`.
        :type parameter_dict: list
        :param forbidden_clauses: list of forbidden parameter configurations, see :doc:`pcs`.
        :type parameter_dict: list
        :param deterministic: whether the function to be minimized contains random components, see :ref:`non-deterministic`.
        :type deterministic: bool
        :param num_train_instances: number of instances used during the configuration/optimization, see :ref:`training_instances`.
        :type num_train_instances: int
        :param num_test_instances: number of instances used for testing/validation, see :ref:`validation`.
        :type num_test_instances: int
        :param num_runs: number of independent SMAC runs.
        :type num_runs: int
        :param num_procs: number SMAC runs that can be executed in paralell
        :type num_procs: int
        :param seed: seed for SMAC's Random Number generator. If int, it is used for the first run, additional runs use consecutive numbers. If list, it specifies a seed for every run.
        :type seed: int/list of ints
        :param mem_limit_function_mb: sets the memory limit for your function (value in MB). ``None`` means no restriction. Be aware that this limit is enforced for each SMAC run separately. So if you have 2 parallel runs, pysmac could use twice that value (and twice the value of mem_limit_smac_mb) in total. Note that due to the creation of the subprocess, the amount of memory available to your function is less than the value specified here. This option exists mainly to prevent a memory usage of 100% which will at least slow the system down.
        :type  mem_limit_function_mb: int
        :param t_limit_function_s: cutoff time for a single function call. ``None`` means no restriction. If optimizing run time, SMAC can choose a shorter cutoff than the provided one for individual runs. If `None` was provided, then there is no cutoff ever!
        """

        self.smac_options['algo-deterministic'] = deterministic
        
        # adjust the number of training instances
        num_train_instances = None if (num_train_instances is None) else int(num_train_instances)
        
        if (num_train_instances is not None):
            if (num_train_instances < 1):
                raise ValueError('The number of training instances must be positive!')
            # check if instance features are provided
            if (train_instance_features is not None):
                # make sure it's the right number of instances
                if (len(train_instance_features) != num_train_instances):
                    raise ValueError("You have to provide features for every training instance!")
                # and the same number of features
                nf = len(train_instance_features[0])
                for feature_vector in  train_instance_features:
                    if (len(feature_vector) != nf):
                        raise ValueError("You have to specify the same number of features for every instance!")
                self.smac_options['feature_file'] = os.path.join(self.working_directory ,'features.dat')
                


        num_procs = int(num_procs)
        pcs_string, parser_dict = pysmac.remote_smac.process_parameter_definitions(parameter_dict)

        # adjust the seed variable
        if isinstance(seed, int):
            seed = list(range(seed, seed+num_runs))
        elif isinstance(seed, list) or isinstance(seed, tuple):
            if len(seed) != num_runs:
                raise ValueError("You have to specify a seed for every run!")
        else:
            raise ValueError("The seed variable could not be properly processed!")
        
        
        self.smac_options['runcount-limit'] = max_evaluations
        if t_limit_function_s is not None:
            self.smac_options['cutoff_time'] = t_limit_function_s
        
        
        # create and fill the pcs file
        with open(self.smac_options['pcs-file'], 'w') as fh:
            fh.write("\n".join(pcs_string + conditional_clauses + forbidden_clauses))
        
        #create and fill the instance files
        tmp_num_instances = 1 if num_train_instances is None else num_train_instances
        with open(self.smac_options['instances'], 'w') as fh:
            for i in range(tmp_num_instances):
                fh.write("id_%i\n"%i)
        
        # create and fill the feature file
        if (train_instance_features is not None):
            with open(self.smac_options['feature_file'], 'w') as fh:
                #write a header
                tmp = ['instance_name'] + list(map(lambda i: 'feature{}'.format(i), range(len(train_instance_features[0]))))
                fh.write(",".join(tmp));
                fh.write("\n");

                # and then the actual features
                for i in range(len(train_instance_features)):
                    tmp = ['id_{}'.format(i)] + ["{}".format(f) for f in train_instance_features[i]]
                    fh.write(",".join(tmp))
                    fh.write("\n");
        

        if num_test_instances is not None:
            # TODO: honor the users values for validation if set, and maybe show a warning on stdout
            self.smac_options['validate-only-last-incumbent'] = True
            self.smac_options['validation'] = True
            self.smac_options['test-instances'] = os.path.join(self.working_directory, 'test_instances.dat')
            with open(self.smac_options['test-instances'],'w') as fh:
                for i in range(tmp_num_instances, tmp_num_instances + num_test_instances):
                    fh.write("id_%i\n"%i)

        # make sure the java executable is callable and up-to-date
        java_executable = self.smac_options.pop('java_executable')
        check_java_version(java_executable)

        timeout_quality = self.smac_options.pop('timeout_quality')


        # create and fill the scenario file
        scenario_fn = os.path.join(self.working_directory,self.smac_options.pop('scenario_fn'))

        
        scenario_options = {'algo', 'algo-exec', 'algoExec',
                            'algo-exec-dir', 'exec-dir', 'execDir','execdir',
                            'deterministic', 'algo-deterministic',
                            'paramfile', 'paramFile', 'pcs-file', 'param-file',
                            'run-obj', 'run-objective', 'runObj', 'run_obj',
                            'intra-obj', 'intra-instance-obj', 'overall-obj', 'intraInstanceObj', 'overallObj', 'overall_obj', 'intra_instance_obj',
                            'algo-cutoff-time', 'target-run-cputime-limit', 'target_run_cputime_limit', 'cutoff-time', 'cutoffTime', 'cutoff_time',    
                            'cputime-limit', 'cputime_limit', 'tunertime-limit', 'tuner-timeout', 'tunerTimeout',
                            'wallclock-limit', 'wallclock_limit', 'runtime-limit', 'runtimeLimit', 'wallClockLimit',
                            'output-dir', 'outputDirectory', 'outdir',
                            'instances', 'instance-file', 'instance-dir', 'instanceFile', 'i', 'instance_file', 'instance_seed_file',
                            'test-instances', 'test-instance-file', 'test-instance-dir', 'testInstanceFile', 'test_instance_file', 'test_instance_seed_file',                            
                            'feature-file', 'instanceFeatureFile', 'feature_file'
                            }
        
        additional_options_fn =scenario_fn[:-4]+'.advanced' 
        with open(scenario_fn,'w') as fh, open(additional_options_fn, 'w') as fg:
            for name, value in list(self.smac_options.items()):
                if name in scenario_options:
                    fh.write('%s %s\n'%(name, value))
                else:
                    fg.write('%s %s\n'%(name,value))
        
        # check that all files are actually present, so SMAC has everything to start
        assert all(map(os.path.exists, [additional_options_fn, scenario_fn, self.smac_options['pcs-file'], self.smac_options['instances']])), "Something went wrong creating files for SMAC! Try to specify a \'working_directory\' and set \'persistent_files=True\'."

        # create a pool of workers and make'em work
        pool = MyPool(num_procs)
        argument_lists = [[scenario_fn, additional_options_fn, s, func, parser_dict, self.__mem_limit_smac_mb, smac_classpath(),  num_train_instances, mem_limit_function_mb, t_limit_function_s, self.smac_options['algo-deterministic'], java_executable, timeout_quality] for s in seed]
        
        pool.map(pysmac.remote_smac.remote_smac_function, argument_lists)
        
        pool.close()
        pool.join()
        
        # find overall incumbent and return it
        
        scenario_dir = os.path.join(self.__out_dir,'.'.join(scenario_fn.split('/')[-1].split('.')[:-1]))
        
        run_incumbents = []
        
        for s in seed:
            fn = os.path.join(scenario_dir, 'traj-run-%i.txt'%s)
            run_incumbents.append(read_trajectory_file(fn)[-1])

        run_incumbents.sort(key = operator.itemgetter("Estimated Training Performance"))

        param_dict = run_incumbents[0]['Configuration']

        for k in param_dict:
            param_dict[k] = parser_dict[k](param_dict[k])

        return( run_incumbents[0]["Estimated Training Performance"], param_dict)
