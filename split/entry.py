from .role import Role

class Entry:
	def __init__(self, location, role):
		'''
		location: locale or remote
		role: a Role object to execute the model
		'''
		self.location = location
		self.role = role
	
	def clone(self):
		return Entry(self.location, self.role.clone())

	def save(self, filepath):
		self.role.save(filepath)
	
	def restore(self, filepath):
		self.role.restore(filepath)
