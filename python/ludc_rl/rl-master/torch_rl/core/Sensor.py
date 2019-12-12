class Sensor(object):
	'''
	A sensor built a view over a world state.
	'''
	def __init__(self):
		pass

	def observe(self,word):
		'''
		Return the view over the world
		Args:
			- word: the world state

		Return:
		    - a view (can be of any type)
		'''
		raise NonImplementedError

	def sensor_space(self):
		'''
		Returns the sensor space i.e the set of possible sensors values

		Return:
		    - the sensor space
		'''
		raise NonImplementedError 
