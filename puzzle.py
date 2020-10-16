class State:
	def __init__(self, state):
		self.state = state
		for i in range(len(state)):
			row = state[i]
			if '*' in row: self.star = (i, row.index('*'))

	def value(self, pos):
		return self.state[pos[0]][pos[1]]
	
	def pos(self, i):
		return self.state[i // len(self.state[0])][i % len(self.state[0])]

	def swap(self, row, col):
		swapState = self.copy()
		swapState[self.star[0]][self.star[1]] = self.state[row][col]
		swapState[row][col] = self.state[self.star[0]][self.star[1]]
		return State(swapState)

	def copy(self):
		copyState = []
		for y in range(len(self.state)):
			row = []
			for x in range(len(self.state[0])):
				row.append(self.value((y, x)))
			copyState.append(row)
		return copyState
	
	def equals(self, state):
		return self.state == state.state

	def DebugPrint(self):
		for row in self.state: print("\t".join(row) + "\n")
	
	def getGoal(self):
		goal = []
		for i in range(len(self.state)):
			row = [str(i * len(self.state[0]) + j + 1) for j in range(len(self.state[0]))]
			goal.append(row)
		goal[-1][-1] = "*"
		return State(goal)
		
	def isGoal(self):
		return self.equals(self.getGoal())
			
	def getNeighbors(self):
		star = self.star
		neighbors = []
		if star[0] > 0: neighbors.append((star[0] - 1, star[1]))
		if star[1] > 0: neighbors.append((star[0], star[1]  - 1))
		if star[0] < len(self.state) - 1: neighbors.append((star[0] + 1, star[1]))
		if star[1] < len(self.state[0]) - 1: neighbors.append((star[0], star[1] + 1))
		return neighbors
	
	def ComputeNeighbors(self):
		return list(map(lambda neighbor: (self.value(neighbor), self.swap(neighbor[0], neighbor[1])), self.getNeighbors()))

def memoize(f):
	cache = {}
	def g(x):
		if x not in cache: cache[x] = f(x)
		return cache[x]
	return g

def LoadFromFile(filepath):
	state = []
	with open(filepath) as f:
		for line in f: state.append(line.rstrip('\n').split('\t'))
	return State(state)

# checks neighbors in order up, left, down, right
def BFS(state):
	frontier, discovered, parents = [state], set(), {state: []}
	while frontier:
		current = frontier.pop()
		discovered.add(current)
		if current.isGoal():
			return parents[current]
		for neighbor in current.ComputeNeighbors():
			if not any(map(neighbor[1].equals, discovered)):
				frontier = [neighbor[1]] + frontier
				parents[neighbor[1]] = [neighbor[0]]
				if parents[current]: parents[neighbor[1]] = parents[current] + [neighbor[0]]

# checks neighbors in order right, down, left, up
def DFS(state):
	frontier, discovered, parents = [state], set(), {state: []}
	while frontier:
		current = frontier.pop()
		discovered.add(current)
		if current.isGoal():
			return parents[current]
		for neighbor in current.ComputeNeighbors():
			if not any(map(neighbor[1].equals, discovered)):
				frontier.append(neighbor[1])
				parents[neighbor[1]] = [neighbor[0]]
				if parents[current]: parents[neighbor[1]] = parents[current] + [neighbor[0]]
		
			
def BidirectionalSearch(state):
	goal = state.getGoal()
	frontier, frontierG = [state], [goal]
	discovered, discoveredG = set(), set()
	parents = {state: [], goal: []}
	while frontier or frontierG:
		current, currentG = frontier.pop(), frontierG.pop()
		discovered.add(current)
		discoveredG.add(currentG)
		for d, dG in [(d, dG) for d in discovered for dG in discoveredG]:
			if d.equals(dG):
				if parents[d] and parents[dG]: return parents[d] + parents[dG]
				elif parents[d]: return parents[d]
				elif parents[dG]: return parents[dG]
				else: return None
		for neighbor in current.ComputeNeighbors():
			if not any(map(neighbor[1].equals, discovered)):
				frontier.append(neighbor[1])
				parents[neighbor[1]] = [neighbor[0]]
				if parents[current]: parents[neighbor[1]] = parents[current] + [neighbor[0]]
		for neighborG in currentG.ComputeNeighbors():
			if not any(map(neighborG[1].equals, discoveredG)):
				frontierG.append(neighborG[1])
				parents[neighborG[1]] = [neighborG[0]]
				if parents[currentG]: parents[neighborG[1]] += parents[currentG]
	

def main():
	state = LoadFromFile('game.txt')
	print(BidirectionalSearch(state))

if __name__ == '__main__':
	main()