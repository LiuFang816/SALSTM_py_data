from __future__ import division, print_function

import pysmac

def rosenbrock_4d (x1,x2,x3,x4):
    """ The 4 dimensional Rosenbrock function as a toy model

    The Rosenbrock function is well know in the optimization community and
    often serves as a toy problem. It can be defined for arbitrary
    dimensions. The minimium is always at x_i = 1 with a function value of
    zero. All input parameters are continuous, but we will pretent that 
    x2, x3, and x4 can only take integral values. The search domain for
    all x's is the interval [-5, 5].
    """

    val=( 100*(x2-x1**2)**2 + (x1-1)**2 + 
          100*(x3-x2**2)**2 + (x2-1)**2 + 
          100*(x4-x3**2)**2 + (x3-1)**2)
    return(val)


parameters=dict(\
                # x1 is a continuous ('real') parameter between -5 and 5.
                # The default value/initial guess is 5.
                x1=('real',       [-5, 5], 5),
                
                # x2 can take only integral values, but range and initial
                # guess are identical to x1.
                x2=('integer',    [-5, 5], 5),

                # x3 is encoded as a categorical parameter. Variables of
                # this type can only take values from a finite set.
                # The actual values can be numeric or strings.
                x3=('categorical',[5, 2, 0, 1, -1, -2, 4, -3, 3, -5, -4], 5),


                # x3 is defined as a so called ordinal parameter. This type
                # is similar to 'categorical', but additionally there exists
                # an ordering among the elements.
                x4=('ordinal',    [-5,-4,-3,-2,-1,0,1,2,3,4,5] , 5),


                )
# Note: the definition of x3 and x4 is only to demonstrate the different
# types of variables pysmac supports. Here these definitions are overly
# complicated for this toy model. For example, the definitions of x2 and
# x3 are equivalent, but the purpose of this example is not to show a 
# realistic use case


# The next step is to create a SMAC_optimizer object
opt = pysmac.SMAC_optimizer()

# Then, call its minimize method with (at least) the three mandatory parameters
value, parameters = opt.minimize(
                rosenbrock_4d, # the function to be minimized
                1000,          # the number of function calls allowed
                parameters)    # the parameter dictionary


# the return value is a tuple of the lowest function value and a dictionary
# containing corresponding parameter setting.
print(('Lowest function value found: %f'%value))
print(('Parameter setting %s'%parameters))
