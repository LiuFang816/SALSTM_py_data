import os

import pysmac.remote_smac

def check_java_version(java_executable="java"):
    """
    Small function to ensure that Java (version >= 7) was found.
    
    As SMAC requires a Java Runtime Environment (JRE), pysmac checks that a
    adequate version (>7) has been found. It raises a RuntimeError
    exception if no JRE or an out-dated version was found.
    
    :param java_executable: callable Java binary. It is possible to pass additional options via this argument to the JRE, e.g. "java -Xmx128m" is a valid argument.
    :type  java_executable: str
    :raises: RuntimeError
    """
    import re
    from subprocess import STDOUT, check_output
    
    error_msg = ""
    error = False
    
    out = check_output(java_executable.split() + ["-version"], stderr=STDOUT).strip().split(b"\n")
    if len(out) < 1:
        error_msg = "Failed checking Java version. Make sure Java version 7 or greater is installed."
        error =  True
    m = re.match(b'.*version "\d+.(\d+)..*', out[0])
    if m is None or len(m.groups()) < 1:
        error_msg = ("Failed checking Java version. Make sure Java version 7 or greater is installed.")
        error = True
    else:
        java_version = int(m.group(1))
        if java_version < 7:
            error_msg = "Found Java version %d, but Java version 7 or greater is required." % java_version
            error = True
    if error:
        raise RuntimeError(error_msg)


def smac_classpath():
    """
    Small function gathering all information to build the java class path.
    
    :returns: string representing the Java classpath for SMAC
    
    """
    import multiprocessing
    from pkg_resources import resource_filename
    
    logger = multiprocessing.get_logger()
    
    smac_folder = resource_filename("pysmac", 'smac/%s' % pysmac.remote_smac.SMAC_VERSION)
    
    smac_conf_folder = os.path.join(smac_folder, "conf")
    smac_patches_folder = os.path.join(smac_folder, "patches")
    smac_lib_folder = os.path.join(smac_folder, "lib")


    classpath = [fname for fname in os.listdir(smac_lib_folder) if fname.endswith(".jar")]
    classpath = [os.path.join(smac_lib_folder, fname) for fname in classpath]
    classpath = [os.path.abspath(fname) for fname in classpath]
    classpath.append(os.path.abspath(smac_conf_folder))
    classpath.append(os.path.abspath(smac_patches_folder))

    # For Windows compability
    classpath = (os.pathsep).join(classpath)

    logger.debug("SMAC classpath: %s", classpath)

    return classpath

