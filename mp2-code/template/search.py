# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,astar)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
from heapq import heappop, heappush

def search(maze, searchMethod):
    return {
        "bfs": bfs,
    }.get(searchMethod, [])(maze)

def backtrace(parent, start, obj):
    path = [obj]
    node = obj
    while node != start:
        node = parent[node]
        path.append(node)
    path.reverse()
    return path

def bfs(maze):
    # Write your code here
    """
    This function returns optimal path in a list, which contains start and objective.
    If no path found, return None. 
    """
    path = []
    frontiers = []
    explored = []
    parent = {}
    start = maze.getStart()
    frontiers.append(start)
    while frontiers:
        node = frontiers.pop(0)
        if maze.isObjective(node[0], node[1], node[2]):
            path = backtrace(parent, start, node)
            return path
        nbs = maze.getNeighbors(node[0], node[1], node[2])
        explored += [node]
        for s in nbs:
            if (s not in explored) and (s not in frontiers):
                frontiers.append(s)
                parent[s] = node
    return None

