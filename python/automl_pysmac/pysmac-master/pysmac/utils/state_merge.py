import os
import glob
import operator
import errno
import filecmp
import shutil
import numpy


from .smac_output_readers import *



def find_largest_file (glob_pattern):
    """ Function to find the largest file matching a glob pattern.

    Old SMAC version keep several versions of files as back-ups. This
    helper can be used to find the largest file (which should contain the
    final output). One could also go for the most recent file, but that
    might fail when the data is copied.

    :param glob_pattern: a UNIX style pattern to apply
    :type glob_pattern: string

    :returns: string -- largest file matching the pattern 
    """
    fns = glob.glob(glob_pattern)

    if len(fns) == 0:
        raise RuntimeError("No file matching pattern \'{}\' found!".format(glob_pattern))

    f_name = ""
    f_size = -1

    for fn in fns:
        s = os.lstat(fn).st_size
        if (s > f_size):
            f_size = s
            f_name = fn
    return(f_name)


def read_sate_run_folder(directory, rar_fn = "runs_and_results-it*.csv",inst_fn = "instances.txt" , feat_fn = "instance-features.txt" , ps_fn = "paramstrings-it*.txt"):    
    """ Helper function that can reads all information from a state_run folder.
    
    To get all information of a SMAC run, several different files have
    to be read. This function provides a short notation for gathering
    all data at once.
    
    :param directory: the location of the state_run_folder
    :type directory: str
    :param rar_fn: pattern to find the runs_and_results file
    :type rar_fn: str
    :param inst_fn: name of the instance file
    :type inst_fn: str
    :param feat_fn: name of the instance feature file. If this file is not found, pysmac assumes no instance features.
    :type feat_fn: str
    :param ps_fn: name of the paramstrings file
    :type ps_fn: str
    
    :returns: tuple -- (configurations returned by read_paramstring_file,\n
        instance names returned by read_instance_file,\n
        instance features returned by read_instance_features_file,\n
        actual run data returned by read_runs_and_results_file)
    """
    print(("reading {}".format(directory)))
    configs = read_paramstrings_file(find_largest_file(os.path.join(directory,ps_fn)))
    instance_names = read_instances_file(find_largest_file(os.path.join(directory,inst_fn)))
    runs_and_results = read_runs_and_results_file(find_largest_file(os.path.join(directory, rar_fn)))

    full_feat_fn = glob.glob(os.path.join(directory,feat_fn))
    if len(full_feat_fn) == 1:      
        instance_features = read_instance_features_file(full_feat_fn[0])
    else:
        instance_features = None

    return (configs, instance_names, instance_features, runs_and_results)



def state_merge(state_run_directory_list, destination, 
                check_scenario_files = True, drop_duplicates = False,
                instance_subset = None):
    """ Function to merge multiple state_run directories into a single
    run to be used in, e.g., the fANOVA.
    
    To take advantage of the data gathered in multiple independent runs,
    the state_run folders have to be merged into a single directory that
    resemble the same structure. This allows easy application of the
    pyfANOVA on all run_and_results files.
    
    :param state_run_directory_list: list of state_run folders to be merged
    :type state_run_directory_list: list of str
    :param destination: a directory to store the merged data. The folder is created if needed, and already existing data in that location is silently overwritten.
    :type destination: str
    :param check_scenario_files: whether to ensure that all scenario files in all state_run folders are identical. This helps to avoid merging runs with different settings. Note: Command-line options given to SMAC are not compared here!
    :type check_scenario_files: bool
    :param drop_duplicates: Defines how to handle runs with identical configurations. For deterministic algorithms the function's response should be the same, so dropping duplicates is safe. Keep in mind that every duplicate effectively puts more weight on a configuration when estimating parameter importance.
    :type drop_duplicates: bool
    :param instance_subset: Defines a list of instances that are used for the merge. All other instances are ignored. (Default: None, all instances are used)
    :type instance_subset: list
    """

    configurations = {}
    instances = {}
    runs_and_results = {}
    ff_header= set()
    
    i_confs = 1;
    i_insts = 1;


    # make sure all pcs files are the same
    pcs_files = [os.path.join(d,'param.pcs') for d in state_run_directory_list]
    if not all([filecmp.cmp(fn, pcs_files[0]) for fn in pcs_files[1:]]):
        raise RuntimeError("The pcs files of the different runs are not identical!")

    #check the scenario files if desired
    scenario_files = [os.path.join(d,'scenario.txt') for d in state_run_directory_list]
    if check_scenario_files and not all([filecmp.cmp(fn, scenario_files[0]) for fn in scenario_files[1:]]):
        raise RuntimeError("The scenario files of the different runs are not identical!")

    for directory in state_run_directory_list:
        try:
            confs, inst_names, tmp , rars = read_sate_run_folder(directory)
            (header_feats, inst_feats) = tmp if tmp is not None else (None,None)
        
        except:
            print(("Something went wrong while reading {}. Skipping it.".format(directory)))
            continue
        
        # confs is a list of dicts, but dicts are not hashable, so they are
        # converted into a tuple of (key, value) pairs and then sorted
        confs = [tuple(sorted(d.items())) for d in confs]        
        
        # merge the configurations
        for conf in confs:
            if not conf in configurations:
                configurations[conf] = {'index': i_confs}
                i_confs += 1
        # merge the instances
        ignored_instance_ids = []
        for i in range(len(inst_names)):
            
            if instance_subset is not None and inst_names[i][0] not in instance_subset:
                ignored_instance_ids.append(i)
                continue
            
            if not inst_names[i][0] in instances:
                instances[inst_names[i][0]] = {'index': i_insts}
                instances[inst_names[i][0]]['features'] =  inst_feats[inst_names[i][0]] if inst_feats is not None else None
                instances[inst_names[i][0]]['additional info'] = ' '.join(inst_names[i][1:]) if len(inst_names[i]) > 1 else None
                i_insts += 1
            else:
                if (inst_feats is None):
                    if not (instances[inst_names[i][0]]['features'] is None):
                        raise ValueError("The data contains the same instance name ({}) twice, but once with and without features!".format(inst_names[i]))
                elif not numpy.all(instances[inst_names[i][0]]['features'] == inst_feats[inst_names[i][0]]):
                    raise ValueError("The data contains the same instance name ({}) twice, but with different features!".format(inst_names[i]))
                pass
        
        # store the feature file header:
        if header_feats is not None:
            ff_header.add(",".join(header_feats))
        
            if len(ff_header) != 1:
                raise RuntimeError("Feature Files not consistent across runs!\n{}".format(header_feats))
        
        
        if len(rars.shape) == 1:
            rars = numpy.array([rars])

        for run in rars:
            # get the local configuration and instance id
            lcid, liid = int(run[0])-1, int(run[1])-1

            if liid in ignored_instance_ids:
                continue

            # translate them into the global ones
            gcid = configurations[confs[lcid]]['index']
            giid = instances[inst_names[liid][0]]['index']

            # check for duplicates and skip if necessary
            if (gcid, giid) in runs_and_results:
                if drop_duplicates:
                    #print('dropped duplicate: configuration {} on instace {}'.format(gcid, giid))
                    continue
                else:
                    runs_and_results[(gcid, giid)].append(run[2:])
            else:
                runs_and_results[(gcid, giid)] = [run[2:]]

    # create output directory
    try:
        os.makedirs(destination)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
        
    # create all files, overwriting existing ones
    shutil.copy(pcs_files[0], destination)
    shutil.copy(scenario_files[0], destination)
        

    with open(os.path.join(destination, 'instances.txt'),'w') as fh:
        sorted_instances = []
        for name in instances:
            if instances[name]['additional info'] is not None:
                sorted_instances.append( (instances[name]['index'], name + ' ' + instances[name]['additional info']) )
            else:
                sorted_instances.append( (instances[name]['index'], name) )
        
        sorted_instances.sort()
        fh.write('\n'.join(map(operator.itemgetter(1), sorted_instances)))
        fh.write('\n')

    with open(os.path.join(destination, 'runs_and_results-it0.csv'),'w') as fh:
        cumulative_runtime = 0.0
        
        fh.write("Run Number,Run History Configuration ID,Instance ID,"
                 "Response Value (y),Censored?,Cutoff Time Used,"
                 "Seed,Runtime,Run Length,"
                 "Run Result Code,Run Quality,SMAC Iteration,"
                 "SMAC Cumulative Runtime,Run Result,"
                 "Additional Algorithm Run Data,Wall Clock Time,\n")
        run_i = 1
        for ((conf,inst),res) in list(runs_and_results.items()):
            for r in res:
                fh.write('{},{},{},'.format(run_i, conf, inst))
                fh.write('{},{},{},'.format(r[0], int(r[1]), r[2]))
                fh.write('{},{},{},'.format(int(r[3]), r[4], r[5]))
                fh.write('{},{},{},'.format(int(r[6]), r[7], 0))
                
                cumulative_runtime += r[4]
                if r[10] == 2:
                    tmp = 'SAT'       
                if r[10] == 1:
                    tmp = 'UNSAT'
                if r[10] == 0:
                    tmp = 'TIMEOUT'
                if r[10] == -1:
                    tmp = 'CRASHED'
                
                fh.write('{},{},,{},'.format(cumulative_runtime,tmp, r[11]))
                fh.write('\n')
                run_i += 1

    with open(os.path.join(destination, 'paramstrings-it0.txt'),'w') as fh:
        sorted_confs = [(configurations[k]['index'],k) for k in list(configurations.keys())]
        sorted_confs.sort()
        for conf in sorted_confs:
            fh.write("{}: ".format(conf[0]))
            fh.write(", ".join(["{}='{}'".format(p[0],p[1]) for p in conf[1]]))
            fh.write('\n')

    
    #print(instances.values())
    
    
    if header_feats is not None:
        with open(os.path.join(destination, 'instance-features.txt'),'w') as fh:
            fh.write("instance," + ff_header.pop())
            sorted_features = [(instances[inst]['index'], inst + ',' + ",".join(list(map(str, instances[inst]['features']))) ) for inst in instances]
            sorted_features.sort()
            fh.write('\n'.join([ t[1] for t in sorted_features]))

    return(configurations, instances, runs_and_results, sorted_instances, sorted_confs, inst_feats)
