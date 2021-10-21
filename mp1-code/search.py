# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,extra)
import sys, queue
from copy import copy, deepcopy

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar":astar,
        "astar_corner": astar_multi,
        "astar_multi": astar_multi,
        "extra": extra,
    }.get(searchMethod)(maze)


def sanity_check(maze, path):
    """
    Runs check functions for part 0 of the assignment.

    @param maze: The maze to execute the search on.
    @param path: a list of tuples containing the coordinates of each state in the computed path

    @return bool: whether or not the path pass the sanity check
    """
    # TODO: Write your code here

    return False

def backtrace(parent, start, obj):
    path = [obj]
    node = obj
    counter = 0
    while node != start and counter<500:
        #print(node)
        #print(start)
        counter += 1
        node = parent[node]
        path.append(node)
    path.reverse()
    return path

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    path = []
    frontiers = []
    explored = []
    parent = {}
    start = maze.getStart()
    objs = maze.getObjectives()
    objs_count = 0
    midpoint = start
    path += [start]
    frontiers.append(start)
    counter = 0
    while frontiers and counter<1000:
        counter += 1
        node = frontiers.pop(0)
        if maze.isObjective(node[0], node[1]):
            subPath = backtrace(parent, midpoint, node)
            midpoint = node
            path += subPath[1:]
            objs_count += 1 
            if objs_count == len(objs):
                return path
        nbs = maze.getNeighbors(node[0], node[1])
        explored += [node]
        for s in nbs:
            if s not in explored:
                frontiers.append(s)
                parent[s] = node
    return path

def backtraceMul(parent, start, obj, curState):
    path = [obj]
    node = obj
    state = curState
    counter = 0 
    while node != start and parent[(node, state)][1] != 0:
        print(node)
        print(state)
        node, state = parent[(node, state)]
        path.append(node)
    path.reverse()
    return path

def manhattan(node_a, node_b):
    if not isinstance(node_a, tuple):
        return "position must be tuple"
    if not isinstance(node_b, tuple):
        return "position must be tuple"
    return abs(node_a[0]-node_b[0]) + abs(node_a[1]-node_b[1])

def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    frontiers = {}
    explored = []
    parent = {}
    start = maze.getStart()
    objs = maze.getObjectives()
    frontiers[start] = manhToNearestObj(start, objs)
    counter = 0 
    while frontiers and counter<1000:
        counter += 1 
        node = min(frontiers, key=frontiers.get)
        frontiers.pop(node)
        if maze.isObjective(node[0], node[1]):
            path = backtrace(parent, start, node)
            return path
        nbs = maze.getNeighbors(node[0], node[1])
        explored += [node]
        for s in nbs:
            if s not in explored:
                parent[s] = node
                frontiers[s] = manhToNearestObj(s, objs) + len(backtrace(parent, start, s)) 
    return []

def astar_dis(maze, start, end):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    frontiers = {}
    explored = []
    parent = {}
    objs = [end]
    frontiers[start] = manhToNearestObj(start, objs)
    counter = 0 
    while frontiers and counter<1000:
        counter += 1 
        node = min(frontiers, key=frontiers.get)
        frontiers.pop(node)
        if node == end:
            path = backtrace(parent, start, node)
            return len(path)
        nbs = maze.getNeighbors(node[0], node[1])
        explored += [node]
        for s in nbs:
            if s not in explored:
                parent[s] = node
                frontiers[s] = manhToNearestObj(s, objs) + len(backtrace(parent, start, s)) 
    return len(path)

def manhToNearestObj(node, objs):
    #manhatton heuristic
    minDis = sys.maxsize
    for o in objs:
        manDis = abs(node[0] - o[0]) + abs(node[1] - o[1])
        if manDis < minDis:
            minDis = manDis
    return minDis

def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    # TODO: Write your code here
    return astar_multi(maze)
    

def heuristicMulObjs(node, objs):
    unvisitedObjs = objs[:]
    currentPos = node
    heuristics = 0
    counter = 0
    while unvisitedObjs and counter<500:
        counter += 1
        distObjs = []
        for o in unvisitedObjs:
            dist = manhattan(currentPos, o)
            distObjs.append((dist, o))
        currentDist, currentCorner = min(distObjs)
        heuristics += currentDist
        currentPos = currentCorner
        unvisitedObjs.remove(currentCorner)
    return heuristics

class State:
    def __init__(self, row, col, heuristic, costsofar, unvisited):
        self.pos = (row, col)
        self.h = heuristic # heuristic
        self.g = costsofar # cost so far 
        self.f = self.h + self.g # f = g + h
        self.unvisitedObjs = deepcopy(unvisited)
        
    def __lt__(self, other):
        return self.f < other.f

def bfs_m(maze):
    # not finished
    frontiers = queue.PriorityQueue()
    startPos = maze.getStart()
    objs_left = maze.getObjectives()
    path = []
    parent = {}
    cost = {}
    visited = []
    parent[startPos, tuple(deepcopy(objs_left))] = None

    startState = State(startPos[0], startPos[1], 0, 0, deepcopy(objs_left))
    frontiers.put(startState)
    counter = 0
    while not frontiers.empty() and counter < 1000:
        counter += 1
    
        curState = frontiers.get()
        curPos = curState.pos 
        if (curPos, curState.unvisitedObjs) in visited:
            continue
        if curPos in objs_left:
            objs_left.remove(curPos)
            if len(objs_left) == 0:
                path = getPath(parent, curPos, [curPos])
                return path

        nbs = maze.getNeighbors(curPos[0], curPos[1])
        cpObjsLeft = deepcopy(objs_left) 
        for s in nbs:
            newCost = curState.g + manhattan(s, curPos)

            if ((s, tuple(cpObjsLeft)) not in cost or cost[(s, tuple(cpObjsLeft))] > newCost):
                newState = State(s[0], s[1], 0, newCost, cpObjsLeft)
                cost[(s, tuple(cpObjsLeft))] = newCost
                frontiers.put(newState)
                parent[(s, tuple(cpObjsLeft))] = (curPos, tuple(deepcopy(curState.unvisitedObjs)))
        visited.append((curPos, cpObjsLeft))
    return []

def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    frontiers = queue.PriorityQueue()
    startPos = maze.getStart()
    objs_left = maze.getObjectives()
    MST_table = {} #[unvisitedObjs]:MST_weights
    path = []
    parent = {}
    cost = {}
    visited = []
    
    parent[startPos, tuple(deepcopy(objs_left))] = None
    # graph for MST using Objs
    edge_weights = {}
    for i in objs_left:
        for j in objs_left:
                edge_weights[(i, j)] = manhattan(i, j)

    MST_table[tuple(objs_left)] = get_MST(objs_left, edge_weights)

    startState = State(startPos[0], startPos[1], MST_table[tuple(objs_left)], 0, deepcopy(objs_left))
    frontiers.put(startState)
    counter = 0
    
    while not frontiers.empty() and counter < 1000:
        counter += 1
        #print(frontiers.qsize())
        curState = frontiers.get()
        #print("curStatePos", curState.pos)
        #print("curSteteObjs", curState.unvisitedObjs)
        #if (curState.pos, tuple(curState.unvisitedObjs)) in visited:
        #    continue
        
        if len(curState.unvisitedObjs) != len(objs_left):
            continue
        curPos = curState.pos 
        #visited[curPos] = curState.f
        
        if curPos in objs_left:
            objs_left.remove(curPos)
            if len(objs_left) == 0:
                path = getPath(parent, curPos, [curPos])
                return path

        nbs = maze.getNeighbors(curPos[0], curPos[1])
        #print("MST", MST_table)
        #print("nbs", nbs)
        cpObjsLeft = deepcopy(objs_left) 
        for s in nbs:
            if tuple(cpObjsLeft) not in MST_table:
                MST_table[tuple(cpObjsLeft)] = get_MST(cpObjsLeft, edge_weights)
    
            newHeuristic =  manhToNearestObj(s, cpObjsLeft) + MST_table[tuple(cpObjsLeft)]
            newCost = curState.g + manhattan(s, curPos)
            
            #newState.prevState = curState
            if ((s, tuple(cpObjsLeft)) not in cost or cost[(s, tuple(cpObjsLeft))] > newCost) and (s, cpObjsLeft) not in visited:
                newState = State(s[0], s[1], newHeuristic, newCost, cpObjsLeft)
                #print("putPos", s)
                #print("putObjsleft", newState.unvisitedObjs)
                cost[(s, tuple(cpObjsLeft))] = newCost
                frontiers.put(newState)
                parent[(s, tuple(cpObjsLeft))] = (curPos, tuple(deepcopy(curState.unvisitedObjs)))
        visited.append((curPos, cpObjsLeft))
        
    return []

def getPath(parent, pos, objs_left):
    '''
    print("p[(6,1), ((6,1), (6,6))]:", parent[((6,1), ((6,1), (6,6)))])
    print("p[(5,1), ((6,1), (6,6))]:", parent[((5,1), ((6,1), (6,6)))])
    print("p[(5,1), ((6,6))]:", parent[((5,1), tuple([(6,6)]))])
    print("p[(6,1), ((6,6))]:", parent[((6,1), tuple([(6,6)]))])
    print("parent", parent[((2, 4), tuple([(1, 1), (1, 6), (6, 1), (6, 6)]))])
    '''
    curPos = pos
    curObjs = objs_left
    path = [curPos]
    counter = 0
    while parent[(curPos, tuple(curObjs))] != None and counter<500:
        #print("Pos", curPos)
        #print("curObjs", curObjs)
        counter += 1
        curPos, curObjs = parent[curPos, tuple(curObjs)]
        path.append(curPos)
    path.reverse()
    print(path)
    return path

def MinKeyExtract(key, visited):
    min = sys.maxsize
    min_vertex = None
    for v in visited.keys():
        if visited[v] == False and key[v] < min:
            min = key[v]
            min_vertex = v
    return min_vertex

def get_MST(goals, edge_weights):
    # Prim's Algo
    MST_weights = 0
    goals_left = goals[:]
    predecessor = {}
    key = {}
    visited = {}
    
    for g in goals_left:
        predecessor[g] = None
        key[g] = sys.maxsize
        visited[g] = False
        
    start_goal = goals_left[0]
    key[start_goal] = 0
    cross_key = deepcopy(key)
    for i in range(len(goals_left)):
        vertex = MinKeyExtract(key, visited)
        visited[vertex] = True
        for g in goals_left:
            if visited[g]==False and key[g]>edge_weights[(vertex, g)] and edge_weights[(vertex, g)]!=0:
                key[g] = edge_weights[(vertex, g)]
                predecessor[g] = vertex
    """
    print("P", predecessor)
    print("K", key)
    print("V", visited)
    """
    num_edges = 0 
    #print("predecessor", predecessor)
    # calculate MST total weights
    for v in predecessor.keys():
        if predecessor[v] != None:
            MST_weights += edge_weights[v, predecessor[v]]
            #print("v, pred, weight", v, predecessor[v], edge_weights[v, predecessor[v]])
            num_edges += 1
    #print("num edges:", num_edges)
    
    return MST_weights

#------------------------------------------------------------------------------------
def extra(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return []
