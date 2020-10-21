import heapq

def storeUnique(C):
	cache = {}
	def g(*args):
		d = args[0]
		for state in cache:
			if cache[state] == d: return state
		s = C(d)
		cache[s] = d
		return s
	return g

@storeUnique
class State:
	def __init__(self, data):
		self.state = data
		self.rows = len(data)
		self.cols = len(data[0])
		for i in range(self.rows):
			row = self.state[i]
			if '*' in row: self.star = (i, row.index('*'))

	def value(self, row, col):
		return self.state[row][col]
	
	def pos(self, i):
		return self.state[i // self.cols][i % self.cols]

	def index(self, value):
		for (row, col) in [(row, col) for row in range(self.rows) for col in range(self.cols)]:
			if self.state[row][col] == value: return (row, col)

	def swap(self, row, col):
		swapState = self.copy()
		swapState[self.star[0]][self.star[1]] = self.state[row][col]
		swapState[row][col] = self.state[self.star[0]][self.star[1]]
		return State(swapState)

	def copy(self):
		return [row[:] for row in self.state]
	
	def equals(self, obj):
		return self.state == obj.state

	def DebugPrint(self):
		for row in self.state: print("\t".join(row))
		print()
	
	def getGoal(self):
		goal = []
		for i in range(self.rows):
			row = [str(i * self.cols + j + 1) for j in range(self.cols)]
			goal.append(row)
		goal[-1][-1] = "*"
		return State(goal)
		
	def isGoal(self):
		for i in range(1, self.rows * self.cols):
			if self.state[(i - 1) // self.cols][(i - 1) % self.cols] != str(i): return False
		return True
	
	def getNeighbors(self):
		neighbors = []
		if self.star[0] > 0: neighbors.append((self.star[0] - 1, self.star[1]))
		if self.star[1] > 0: neighbors.append((self.star[0], self.star[1]  - 1))
		if self.star[0] < self.rows - 1: neighbors.append((self.star[0] + 1, self.star[1]))
		if self.star[1] < self.cols - 1: neighbors.append((self.star[0], self.star[1] + 1))
		return neighbors
	
	def ComputeNeighbors(self):
		return list(map(lambda pos: (self.value(pos[0], pos[1]), self.swap(pos[0], pos[1])), self.getNeighbors()))

# memoizes when state is equal to one in cache
def memoize(f):
	cache = {}
	def g(x):
		if x in cache: return cache[x]
		cache[x] = f(x)
		return f(x)
	return g

def isSolvable(state):
	inversions = 0
	it = [i for i in range(state.cols * state.rows) if i != state.star[0] * state.cols + state.star[1]]
	greatest = (int(state.state[it[-1] // state.cols][it[-1] % state.cols]) + 1, 0, -1)
	for iterator in range(len(it) - 2, -1, -1):
		num = int(state.state[it[iterator] // state.cols][it[iterator] % state.cols])
		base = (0, 0, None)
		if greatest[0] + 1 == num: base = greatest
		inv = base[1]
		for i in it[iterator+1:base[2]]:
			comp = int(state.state[i // state.cols][i % state.cols])
			if num > comp: inv += 1
		inversions += inv
		if  num > greatest[0]: greatest = (num, inv + 1, iterator)
	if state.cols % 2 == 0 and (state.rows - state.star[0]) % 2 == 0:	return inversions % 2 == 1
	return inversions % 2 == 0			

def swapValues(state, l):
	newState = state
	for v in l:
		pos = newState.index(v)
		newState = newState.swap(pos[0], pos[1])
	return newState

def LoadFromFile(filepath):
	state = []
	with open(filepath) as f:
		for line in f: state.append(line.rstrip('\n').split('\t'))
	return State(state)

# checks neighbors in order up, left, down, right
def BFS(state):
	if not isSolvable(state) or state.isGoal(): return []
	frontier, discovered, parents = [], [state], {}
	for neighbor in state.ComputeNeighbors():
		frontier = [neighbor[1]] + frontier
		discovered.append(neighbor[1])
		parents[neighbor[1]] = [neighbor[0]]
	while frontier:
		current = frontier.pop()
		if current.isGoal(): return parents[current]
		for neighbor in current.ComputeNeighbors():
			if neighbor[1] not in discovered:
				frontier = [neighbor[1]] + frontier
				discovered.append(neighbor[1])
				parents[neighbor[1]] = parents[current] + [neighbor[0]]

# checks neighbors in order right, down, left, up
def DFS(state):
	if not isSolvable(state) or state.isGoal(): return []
	frontier, discovered, parents = [], [state], {}
	for neighbor in state.ComputeNeighbors():
		frontier.append(neighbor[1])
		discovered.append(neighbor[1])
		parents[neighbor[1]] = [neighbor[0]]
	while frontier:
		current = frontier.pop()
		if current.isGoal(): return parents[current]
		for neighbor in current.ComputeNeighbors():
			if neighbor[1] not in discovered:
				frontier.append(neighbor[1])
				discovered.append(neighbor[1])
				parents[neighbor[1]] = parents[current] + [neighbor[0]]
		
			
def BidirectionalSearch(state):
	if not isSolvable(state): return []
	goal = state.getGoal()
	if state.equals(goal): return []
	frontier, discovered, parents = [], [state], {}
	for neighbor in state.ComputeNeighbors():
		frontier = [neighbor[1]] + frontier
		discovered.append(neighbor[1])
		parents[neighbor[1]] = [neighbor[0]]
	frontierG, discoveredG, parentsG = [], [goal], {}
	for neighborG in goal.ComputeNeighbors():
		frontierG = [neighborG[1]] + frontierG
		discoveredG.append(neighborG[1])
		parentsG[neighborG[1]] = [neighborG[0]]
	while frontier or frontierG:
		current, currentG = frontier.pop(), frontierG.pop()
		if current in discoveredG: return parents[current] + parentsG[current]
		if currentG in discovered: return parents[currentG] + parentsG[currentG]
		for neighbor in current.ComputeNeighbors():
			if neighbor[1] not in discovered:
				frontier = [neighbor[1]] + frontier
				discovered.append(neighbor[1])
				parents[neighbor[1]] = parents[current] + [neighbor[0]]
		for neighborG in currentG.ComputeNeighbors():
			if neighborG[1] not in discoveredG:
				frontierG = [neighborG[1]] + frontierG
				discoveredG.append(neighborG[1])
				parentsG[neighborG[1]] = [neighborG[0]] + parentsG[currentG]

def AStar(state):
	if not isSolvable(state) or state.isGoal(): return []
	count = 0
	frontier, discovered, parents = [], [state], {}
	for neighbor in state.ComputeNeighbors():
		t = (totalDisplacement(neighbor[1]), count, neighbor[1])
		heapq.heappush(frontier, t)
		discovered.append(neighbor[1])
		parents[neighbor[1]] = [neighbor[0]]
		count += 1
	while frontier:
		current = heapq.heappop(frontier)
		if current[2].isGoal():
			return parents[current[2]]
		for neighbor in current[2].ComputeNeighbors():
			if neighbor[1] not in discovered:
				t = (current[0] + totalDisplacement(neighbor[1]), count, neighbor[1])
				heapq.heappush(frontier, t)
				discovered.append(neighbor[1])
				parents[neighbor[1]] = parents[current[2]] + [neighbor[0]]
				count += 1
	
@memoize
def totalDisplacement(state):
	dis = 0
	omit = state.star[0] * state.cols + state.star[1]
	for i in range(omit):
		val = int(state.state[i // state.cols][i % state.cols])
		correctPos = ((val - 1) // state.cols, (val - 1) % state.cols)
		dis += abs(correctPos[0] - (i // state.cols)) + abs(correctPos[1] - (i % state.cols))
	dis += abs(state.rows - state.star[0] - 1) + abs(state.cols - state.star[1] - 1)
	for i in range(omit + 1, state.cols * state.rows):
		val = int(state.state[i // state.cols][i % state.cols])
		correctPos = ((val - 1) // state.cols, (val - 1) % state.cols)
		dis += abs(correctPos[0] - (i // state.cols)) + abs(correctPos[1] - (i % state.cols))
	return dis

def main():
	state = LoadFromFile('game.txt')
	f = BFS
	print(str(f.__name__) + ": " + (str(f(state))))

if __name__ == '__main__':
	main()