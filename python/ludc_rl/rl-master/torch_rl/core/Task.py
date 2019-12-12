class Task(object):
	'''
	A Task  defines a particular learning task (including reward-based problems, but not only). A Task also defines when an episode finished.
	'''
	def __init__(self):
		pass

	def finished(self,world):
		'''
		Tell if the world is in a final state (no more action can be applied)

		Args:
			-world: the world state to test

		Return:
		     - True if the word is in a final state, False elsewhere
		'''
		raise NonImplementedError

