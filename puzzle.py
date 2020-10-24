from collections import deque
import heapq

def memoizeClass(C):
	cache = {}
	def constructUnique(*args):
		s = str(args[0])
		if s in cache: return cache[s]
		cache[s] = C(*args)
		return cache[s]
	return constructUnique

@memoizeClass
class State:
	def __init__(self, data, star):
		self.state = data
		self.star = star
		self.rows = len(data)
		self.cols = len(data[0])

	def pos(self, i):
		return self.state[i // self.cols][i % self.cols]

	def index(self, value):
		for (row, col) in [(row, col) for row in range(self.rows) for col in range(self.cols)]:
			if self.state[row][col] == value: return (row, col)

	def swap(self, row, col):
		swapState = self.copy()
		swapState[self.star[0]][self.star[1]] = self.state[row][col]
		swapState[row][col] = '*'
		return State(swapState, (row, col))

	def copy(self):
		return [row[:] for row in self.state]

	def __gt__(self, other):
		return False

	def getGoal(self):
		goal = []
		for i in range(self.rows):
			row = [str(i * self.cols + j + 1) for j in range(self.cols)]
			goal.append(row)
		goal[-1][-1] = '*'
		return State(goal, (-1, -1))
		
	def isGoal(self):
		for i in range(1, self.rows * self.cols):
			if self.state[(i - 1) // self.cols][(i - 1) % self.cols] != str(i): return False
		return True
		
	def recursManhattanDistance(self, val):
		pos = self.index(val)
		val = int(val)
		goalRow = (val - 1) // self.cols
		goalCol = (val - 1) % self.cols
		newDistance = abs(self.star[0] - goalRow) + abs(self.star[1] - goalCol)
		oldDistance = abs(pos[0] - goalRow) + abs(pos[1] - goalCol)
		return newDistance - oldDistance

	def recursManhattanDistance2(self, pos):
		val = int(self.state[pos[0]][pos[1]])
		goalRow = (val - 1) // self.cols
		goalCol = (val - 1) % self.cols
		newDistance = abs(self.star[0] - goalRow) + abs(self.star[1] - goalCol)
		oldDistance = abs(pos[0] - goalRow) + abs(pos[1] - goalCol)
		return newDistance - oldDistance

	def getNeighbors(self):
		neighbors = []
		if self.star[0] < self.rows - 1: neighbors.append((self.star[0] + 1, self.star[1]))
		if self.star[1] < self.cols - 1: neighbors.append((self.star[0], self.star[1] + 1))
		if self.star[0] > 0: neighbors.append((self.star[0] - 1, self.star[1]))
		if self.star[1] > 0: neighbors.append((self.star[0], self.star[1]  - 1))
		return neighbors
	
	def ComputeNeighbors(self):
		return [(self.state[pos[0]][pos[1]], self.swap(pos[0], pos[1])) for pos in self.getNeighbors()]

	def DebugPrint(self):
		for row in self.state: print("\t".join(row))
		print()

	def testSwaps(self, l):
		state = self
		for v in l:
			pos = state.index(v)
			state = state.swap(pos[0], pos[1])
		return state

# memoizes when state is equal to one in cache
def memoize(f):
	cache = {}
	def g(x):
		if x in cache: return cache[x]
		cache[x] = f(x)
		return cache[x]
	return g

def isSolvable(state):
	inversions = 0
	omit = state.star[0] * state.cols + state.star[1]
	it = [i for i in range(omit)] + [i for i in range(omit + 1, state.rows * state.cols)]
	greatest = (int(state.state[it[-1] // state.cols][it[-1] % state.cols]), 1, -1)
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

def LoadFromFile(filepath):
	data = []
	star = ()
	with open(filepath) as f:
		for line in f: data.append(line.rstrip('\n').split('\t'))
	for i in range(len(data)):
			row = data[i]
			if '*' in row: star = (i, row.index('*'))
	return State(data, star)

# checks neighbors in order up, left, down, right
def BFS(state):
	if not isSolvable(state) or state.isGoal(): return []
	frontier, discovered, parents = deque([]), [state], {}
	for neighbor in state.ComputeNeighbors():
		frontier.append(neighbor[1])
		discovered.append(neighbor[1])
		parents[neighbor[1]] = [neighbor[0]]
	while frontier:
		current = frontier.popleft()
		if current.isGoal(): return parents[current]
		for neighbor in current.ComputeNeighbors():
			if neighbor[1] not in discovered:
				frontier.append(neighbor[1])
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
	if state == goal: return []
	frontier, discovered, parents = deque([]), [state], {}
	for neighbor in state.ComputeNeighbors():
		frontier.append(neighbor[1])
		discovered.append(neighbor[1])
		parents[neighbor[1]] = [neighbor[0]]
	frontierG, discoveredG, parentsG = deque([]), [goal], {}
	for neighborG in goal.ComputeNeighbors():
		frontierG.append(neighborG[1])
		discoveredG.append(neighborG[1])
		parentsG[neighborG[1]] = [neighborG[0]]
	while frontier or frontierG:
		current, currentG = frontier.popleft(), frontierG.popleft()
		if current in discoveredG: return parents[current] + parentsG[current]
		if currentG in discovered: return parents[currentG] + parentsG[currentG]
		for neighbor in current.ComputeNeighbors():
			if neighbor[1] not in discovered:
				frontier.append(neighbor[1])
				discovered.append(neighbor[1])
				parents[neighbor[1]] = parents[current] + [neighbor[0]]
		for neighborG in currentG.ComputeNeighbors():
			if neighborG[1] not in discoveredG:
				frontierG.append(neighborG[1])
				discoveredG.append(neighborG[1])
				parentsG[neighborG[1]] = [neighborG[0]] + parentsG[currentG]

def manhattanDistance(state):
	dis = 0
	data = state.copy()
	data[state.star[0]][state.star[1]] = 0
	data = [[int(item) for item in row] for row in data]
	for row in range(state.rows):
		for col in range(state.cols):
			val = data[row][col]
			if val:
				correctPos = ((val - 1) // state.cols, (val - 1) % state.cols)
				dis += abs(correctPos[0] - row) + abs(correctPos[1] - col)
	return dis

def getConflictTiles(state):
	data = state.copy()
	data[state.star[0]][state.star[1]] = 0
	data = [[int(item) for item in row] for row in data]
	rowConflicts, colConflicts = {}, {}
	rowConflictTiles, colConflictTiles = set(), set()
	for row in range(len(data)):
		conflictsHeap = []
		for tile in range(len(data[0])):
			if data[row][tile]:
				rowConflicts[(tile, row)] = findRowConflict(data, tile, row)
				heapq.heappush(conflictsHeap, (-len(rowConflicts[(tile, row)]), tile))
		while any([rowConflicts[key] for key in rowConflicts if key[1] == row]):
			biggestConflictTile = heapq.heappop(conflictsHeap)[1]
			rowConflicts[(biggestConflictTile, row)] = 0
			for t in conflictsHeap:
				if biggestConflictTile in rowConflicts[(t[1], row)]:
					rowConflicts[(t[1], row)] = [item for item in rowConflicts[(t[1], row)] if item != biggestConflictTile]
			rowConflictTiles.add(str(data[row][biggestConflictTile]))
	for col in range(len(data[0])):
		conflictsHeap = []
		for tile in range(len(data)):
			if data[tile][col]:
				colConflicts[(tile, col)] = findColConflict(data, tile, col)
				heapq.heappush(conflictsHeap, (-len(colConflicts[(tile, col)]), tile))
		while any([colConflicts[key] for key in colConflicts if key[1] == col]):
			biggestConflictTile = heapq.heappop(conflictsHeap)[1]
			colConflicts[(biggestConflictTile, col)] = 0
			for t in conflictsHeap:
				if biggestConflictTile in colConflicts[(t[1], col)]:
					colConflicts[(t[1], col)] = [item for item in colConflicts[(t[1], col)] if item != biggestConflictTile]
			colConflictTiles.add(str(data[biggestConflictTile][col]))
	return (rowConflictTiles, colConflictTiles, 2 * (len(rowConflictTiles) + len(colConflictTiles)))

def findRowConflict(data, tile, row):
	val = data[row][tile]
	if (val - 1) // len(data[0]) != row: return []
	correctCol = (val - 1) % len(data[0])
	if correctCol == tile: return []
	conflictCols = []
	if correctCol > tile:
		for i in range(tile + 1, correctCol + 1):
			if (data[row][i] - 1) // len(data[0]) == row: conflictCols.append(i)
	for i in range(correctCol, tile):
		if (data[row][i] - 1) // len(data[0]) == row: conflictCols.append(i)
	return conflictCols

def findColConflict(data, tile, col):
	val = data[tile][col]
	if (val - 1) % len(data[0]) != col: return []
	correctRow = (val - 1) // len(data[0])
	if correctRow == tile: return []
	conflictRows = []
	if correctRow > tile:
		for i in range(tile + 1, correctRow + 1):
			if (data[i][col] - 1) % len(data[0]) == col: conflictRows.append(i)
	for i in range(correctRow, tile):
		if (data[i][col] - 1) % len(data[0]) == col: conflictRows.append(i)
	return conflictRows

def lastMove(state):
	if str(state.rows * state.cols - 1) not in state.state[-1] and str(state.cols * (state.rows - 1)) not in [row[-1] for row in state.state]: return 2
	return 0

def AStar(state):
	if not isSolvable(state) or state.isGoal(): return []
	frontier, discovered, parents = [], [state], {}
	conflictTiles = getConflictTiles(state)
	prevCost = manhattanDistance(state) + conflictTiles[2]
	for neighbor in state.ComputeNeighbors():
		discovered.append(neighbor[1])
		parents[neighbor[1]] = [neighbor[0]]
		cost = prevCost + 1 + state.recursManhattanDistance(neighbor[0])
		if neighbor[0] in conflictTiles[0]:	cost -= 2
		if neighbor[0] in conflictTiles[1]:	cost -= 2
		t = (cost, neighbor[1])
		heapq.heappush(frontier, t)
	while frontier:
		current = heapq.heappop(frontier)
		prevCost = current[0]
		print(prevCost)
		current = current[1]
		if current.isGoal():
			return parents[current]
		for neighbor in current.ComputeNeighbors():
			if neighbor[1] not in discovered:
				discovered.append(neighbor[1])
				parents[neighbor[1]] = parents[current] + [neighbor[0]]
				cost = prevCost + 1 + current.recursManhattanDistance(neighbor[0])
				if neighbor[0] not in parents[current]:
					if neighbor[0] in conflictTiles[0]: cost -= 2
					if neighbor[0] in conflictTiles[1]: cost -= 2
				t = (cost, neighbor[1])
				heapq.heappush(frontier, t)

#AStar but not admissible
def PessimisticAStar(state):
	if not isSolvable(state) or state.isGoal(): return []
	frontier, discovered, parents = [], [state], {}
	conflictTiles = getConflictTiles(state)
	for neighbor in state.ComputeNeighbors():
		discovered.append(neighbor[1])
		parents[neighbor[1]] = [neighbor[0]]
		cost = state.recursManhattanDistance(neighbor[0])
		if neighbor[0] in conflictTiles[0]:	cost -= 2
		if neighbor[0] in conflictTiles[1]:	cost -= 2
		t = (cost, neighbor[1])
		heapq.heappush(frontier, t)
	while frontier:
		current = heapq.heappop(frontier)
		prevCost = current[0]
		current = current[1]
		if current.isGoal():
			return parents[current]
		for neighbor in current.ComputeNeighbors():
			if neighbor[1] not in discovered:
				discovered.append(neighbor[1])
				parents[neighbor[1]] = parents[current] + [neighbor[0]]
				cost = prevCost + current.recursManhattanDistance(neighbor[0])
				if neighbor[0] not in parents[current]:
					if neighbor[0] in conflictTiles[0]: cost -= 2
					if neighbor[0] in conflictTiles[1]: cost -= 2
				t = (cost, neighbor[1])
				heapq.heappush(frontier, t)

def main():
	state = LoadFromFile('game.txt')
	f = AStar
	print(str(f.__name__) + ": " + (str(f(state))))

if __name__ == '__main__':
	main()