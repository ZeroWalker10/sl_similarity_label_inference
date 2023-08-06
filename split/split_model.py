from .entry import Entry

class SplitModel:
	def __init__(self):
		self.layers = []
	
	def split(self, cut_layers):
		# cur_layers is a list of tuple which contains
		# cut layer configure. e.g. [(5, 'client', device), (9, 'server', device), ...]
		raise "the split method must be implemented in derived class"

	def layer_depth(self):
		return len(self.layers)
