from .entry import Entry

class Pipeline:
	def __init__(self, entries):
		'''
		entries: a Entry list which record all the Roles to be executed
		'''
		self.entries = entries
		self.cur_step = 0

	def reset(self):
		self.cur_step = 0
	
	def r_reset(self):
		self.cur_step = len(self.entries) - 1
	
	def next(self):
		step = self.cur_step
		self.cur_step += 1
		return self.entries[step]
	
	def r_next(self):
		step = self.cur_step
		self.cur_step -= 1
		return self.entries[step]
	
	def is_end(self):
		return self.cur_step >= len(self.entries)
	
	def r_is_end(self):
		return self.cur_step < 0

	def locate_step(self, k):
		self.cur_step = k
		return self.entries[k]
	
	def steps(self):
		return len(self.entries) 
	
	def push(self, entry):
		self.entries.append(entry)
	
	def clone(self):
		entries = []
		for entry in self.entries:
			entries.append(entry.clone())
		return Pipeline(entries)
