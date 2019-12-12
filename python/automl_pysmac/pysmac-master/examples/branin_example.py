from __future__ import print_function, division

import sys
sys.path.append("..")

import pysmac
import math

# To demonstrate the use, we shall look at a slight modification of the well-
# known branin funtcion. To make things a little bit more interesting, we add
# a third parameter (x3). It has to be an integer, which is usually not
# covered by standard global minimizers.
# SMAC will acutally not impress for this function as SMAC's focus is more on
# high dimensional problems with also categorical parameters, and possible
# dependencies between them. This is where the underlying random forest realy
# shines.
def modified_branin(x1, x2, x3):
    # enforce that x3 has an integer value
    if (int(x3) != x3):
        raise ValueError("parameter x3 has to be an integer!")
    a = 1
    b = 5.1 / (4*math.pi**2)
    c = 5 / math.pi
    r = 6
    s = 10
    t = 1 / (8*math.pi)
    ret  = a*(x2-b*x1**2+c*x1-r)**2+s*(1-t)*math.cos(x1)+s + x3
    return ret



# Now we have to define the parameters for the function we like to minimize.
# The representation tries to stay close to the SMAC manual, but deviates
# if necessary. # Because we use a Python dictionary to represent the
# parameters, there are different ways of creating it. The author finds the
# following way most intuitive:
parameter_definition=dict(\
                x1=('real',    [-5, 5],  1),     # this line means x1 is a float between -5 and 5, with a default of 1
                x2=('real',    [-5, 5], -1),     # same as x1, but the default is -1
                x3=('integer', [0, 10],  1),     # integer that ranges between 0 and 10, default is 1
                )


# Consult the docs for a comprehensive presentation of possibilities.



# The next step is to create a SMAC_optimizer object
opt = pysmac.SMAC_optimizer(debug=False)

# Then, call its minimize method with at least the three mandatory parameters
value, parameters = opt.minimize(modified_branin,      # the function to be minimized
                                 1000,                 # the maximum number of function evaluations
                                 parameter_definition) # the parameter dictionary


# the return value is a tuple of the lowest function value and a dictionary
# containing corresponding parameter setting.
print(('Lowest function value found: %f'%value))
print(('Parameter setting %s'%parameters))
