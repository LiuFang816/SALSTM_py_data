from __future__ import print_function, division


import pysmac
import numpy
import sklearn.ensemble
import sklearn.datasets
import sklearn.cross_validation


# We use the same data as the earlier sklearn example.
X,Y = sklearn.datasets.make_classification(1000, n_features=20, n_informative=10, n_classes=10, random_state=2)		# seed yields a mediocre initial accuracy on my machine

# But this time, we do not split it into train and test data set, but we will use
# k-fold cross validation instead to estimate the accuracy better. Here,we shall
# use k=10 for demonstration purposes. To make thins more convinient later on,
# let's convert the KFold iterator into a list, so we can use indexing.
kfold = [(train,test) for (train,test) in sklearn.cross_validation.KFold(X.shape[0], 10)]

# To demonstrate the use of features, let's use the class frequencies of a fold
# as features. It will turn out that those are not informative features, as they are almost
# all the identical, but good dataset features are beyond the scope of this example.
features = numpy.array([numpy.bincount(Y[test], minlength=10) for (test,train) in kfold])

# We have to make a slight modification to the function fitting the random forest.
# it now has to take an additional argument instance. (Note: SMAC grew historically
# in the context of algorithm configuration, where the performance across multiple
# instances is optimized. The naming convention is a tribute to that heritage.)
# This argument will be a integer between 0 and num_instances (defined below).
# Note that this increases the computational effort as SMAC now estimates the 
# quality of a parameter setting for multiple instances.
def random_forest(n_estimators,criterion, max_features, max_depth, bootstrap, instance):

	# Use the requested fold
	train, test = kfold[instance]
	X_train, Y_train, X_test, Y_test = X[train], Y[train], X[test], Y[test]
	
	predictor = sklearn.ensemble.RandomForestClassifier(n_estimators = n_estimators, criterion=criterion, max_features = max_features, max_depth = max_depth, bootstrap=bootstrap)
	predictor.fit(X_train, Y_train)
	
	return -predictor.score(X_test, Y_test)


# Convenience function to model compute the true mean accuracy across all
# 10 folds.
def true_accuracy(**config):
	accuracy = 0.

	predictor = sklearn.ensemble.RandomForestClassifier(**config)
	for train, test in kfold:
		X_train, Y_train, X_test, Y_test = X[train], Y[train], X[test], Y[test]
		predictor.fit(X_train, Y_train)
		accuracy += predictor.score(X_test, Y_test)
	return(accuracy/len(kfold))

print('The default accuracy is %f'%true_accuracy())
	

# We haven't changed anything here.
parameter_definition=dict(\
		max_depth   =("integer", [1, 10],  4),
		max_features=("integer", [1, 20], 10),
		n_estimators=("integer", [1,100], 10, 'log'),			
		criterion   =("categorical", ['gini', 'entropy'], 'entropy'),
		bootstrap   =("integer", [0,1], 1)
		)

# Same creation of the SMAC_optimizer object
opt = pysmac.SMAC_optimizer( working_directory = '/tmp/pysmac_test/',# the folder where SMAC generates output
							 persistent_files=True,				 # whether the output will persist beyond the python object's lifetime
							 debug = False							 # if something goes wrong, enable this for diagnostic output
							)


# The minimize method also has optional arguments
value, parameters = opt.minimize(random_forest,
					200, parameter_definition,
					num_runs = 2,					# number of independent SMAC runs
					seed = 0,						# the random seed used. can be an int or a list of ints of length num_runs
					num_procs = 2,					# pysmac can harness multicore architecture. Specify the number of processes to use here.
					num_train_instances = len(kfold),# This tells SMAC how many different instances there are.
					train_instance_features = features# use the features defined above to better predict the overall performance
					)
	
print('Parameter setting %s'%parameters)
print('The highest accuracy estimation: %f'%(-value))
print('The highest accuracy actually is: %f'%(true_accuracy(**parameters)))
