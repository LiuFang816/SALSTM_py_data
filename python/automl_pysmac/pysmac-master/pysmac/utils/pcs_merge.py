import re

def merge_configuration_spaces(*args, **kwargs):
	"""
	Convenience function to merge several algorithms with their respective config spaces into a single one.
	
	Using pySMAC to optimize the parameters of a single function/algorithm
	is a very important usecase, but finding the best algorithm and its
	configuration across multiple choices. For example, when multiple different
	algorithms can be used to solve the same problem, and it is unclear
	which one will be the best choice.
	
	The arguments to this function is a number of tuples, each with the
	following content: (callable, pcs, conditionals, forbiddens). The
	callable is a valid python function in the current namespace. The
	Parameter Configuration Space (pcs) is the definition of its
	parameters, while conditionals and forbiddens express dependencies and
	restrictions within that space.
	
	:returns: the merged configuration space, a list of conditionals, a list of forbiddens, and a string that defines two functions (see :ref:`merge_pcs`).
	"""

	names = []
	parameters = {}
	conditionals = []
	forbiddens = []
	
	# go through all the provided arguments
	for (callabl, params, conds, forbs) in args:
		# store callable's name
		name = callabl.__name__
		names.append(name)
		
		# modify the parameter names
		for p in params:
			parameters[name + '_' + p] = params[p]
		
		# modify existing conditionals
		already_conditioned = set()
		for c in conds:
			c = c.strip(' ')
			already_conditioned.update( (c.split('|')[0].strip(' '),))
			
			for p in params:
				c = (re.subn( p, name + '_' + p , c)[0])
			
			conditionals.append( c.rstrip() + ' && algorithm == '+name)
	
		# add new conditionals
		for p in params:
			if p not in already_conditioned:
				conditionals.append(name + '_'  + p + ' | algorithm == ' + name)
	
		for f in forbs:
			f = f.strip()
			for p in params:
				f = (re.subn( p, name + '_' + p , f)[0])
			forbiddens.append(f)
	parameters['algorithm'] = ('categorical', names, names[0])
	
	wrapper_str="""import re

def pysmac_merged_pcs_reduce_args(algorithm=None, *args, **kwds):
	# create a set of algorithm names to filter out the inactive parameters
	names = %s"""%(names) + '\n\tcallables = [' + ",".join(names) + """]

	# remove unused arguments
	new_kwds = {}
	for p in kwds:
		is_algorithm_parameter = False
		for a in names:
			l = re.split(a+'_', p, maxsplit=1)
			if len(l) == 2:
				is_algorithm_parameter = True
				if a == algorithm:
					new_kwds[l[1]] = kwds[p]
				break
		if not is_algorithm_parameter:
			new_kwds[p] = kwds[p]

	i = names.index(algorithm)
	return((callables[i], new_kwds))


def pysmac_merged_pcs_wrapper(algorithm=None, *args, **kwds):
	
	call, args = pysmac_merged_pcs_reduce_args(algorithm = algorithm, *args, **kwds)

	return(call(**args))"""
		
	return(parameters, conditionals, forbiddens, wrapper_str)
