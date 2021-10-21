# maze.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Michael Abir (abir2@illinois.edu) and 
#            Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018
"""
This file contains the Maze class, which reads in a maze file and creates
a representation of the maze that is exposed through a simple interface.
"""

import copy
from const import *
from util import *

class Maze:
    # Initializes the Maze object by reading the maze from a file
    def __init__(self, input_map, offsets, granularity):        
        self.__start = None
        self.__objective = []        

        self.offsets = offsets
        self.granularity = granularity
    
        self.__dimensions = [len(input_map), len(input_map[0]), len(input_map[0][0])]        
        self.__map = input_map
        for x in range(self.__dimensions[ALPHA]):
            for y in range(self.__dimensions[BETA]):
                for z in range(self.__dimensions[GAMMA]): 
                    if self.__map[x][y][z] == START_CHAR:                    
                        self.__start = idxToAngle((x, y, z), self.offsets, granularity)
                    elif self.__map[x][y][z] == OBJECTIVE_CHAR:
                        self.__objective.append(idxToAngle((x, y, z), self.offsets, granularity))

        if not self.__start:
            print("Maze has no start")            
            raise SystemExit

        if not self.__objective:
            print("Maze has no objectives")
            raise SystemExit

    def getChar(self, alpha, beta, gamma):
        x, y, z = angleToIdx((alpha, beta, gamma), self.offsets, self.granularity)
        return self.__map[x][y][z]

    # Returns True if the given position is the location of a wall
    def isWall(self, alpha, beta, gamma):
        return self.getChar(alpha, beta, gamma) == WALL_CHAR

    # Rturns True if the given position is the location of an objective
    def isObjective(self, alpha, beta, gamma):
        return self.getChar(alpha, beta, gamma) == OBJECTIVE_CHAR

    # Returns the start position as a tuple of (beta, column)
    def getStart(self):
        return self.__start

    def setStart(self, start):
        self.__start = start

    # Returns the dimensions of the maze as a (beta, column) tuple
    def getDimensions(self):
        return self.__dimensions

    # Returns the list of objective positions of the maze
    def getObjectives(self):
        return copy.deepcopy(self.__objective)

    def setObjectives(self, objectives):
        self.__objective = objectives

    # Check if the agent can move into a specific beta and column
    def isValidMove(self, alpha, beta, gamma):
        x, y, z = angleToIdx((alpha, beta, gamma), self.offsets, self.granularity)
        return x >= 0 and x < self.getDimensions()[ALPHA] and \
               y >= 0 and y < self.getDimensions()[BETA] and \
               z >= 0 and z < self.getDimensions()[GAMMA] and \
               not self.isWall(alpha, beta, gamma)
        
    # Returns list of neighboing squares that can be moved to from the given beta,gamma
    def getNeighbors(self, alpha, beta, gamma):
        possibleNeighbors = [
            (alpha + self.granularity, beta, gamma),
            (alpha - self.granularity, beta, gamma),
            (alpha, beta + self.granularity, gamma),
            (alpha, beta - self.granularity, gamma),
            (alpha, beta, gamma + self.granularity),
            (alpha, beta, gamma - self.granularity),            
        ]
        neighbors = []
        for a, b, r in possibleNeighbors:
            if self.isValidMove(a, b, r):
                neighbors.append((a, b, r))
        return neighbors

    def saveToFile(self, filename):        
        outputMap = ""
        for gamma in range(self.__dimensions[2]):
            for beta in range(self.__dimensions[1]):
                for alpha in range(self.__dimensions[0]):
                    outputMap += self.__map[alpha][beta][gamma]
                outputMap += "\n"

        with open(filename, 'w') as f:
            f.write(outputMap)

        return True
            
    def isValidPath(self, path):
        # First, check whether it moves single hop
        for i in range(1, len(path)):
            prev = path[i-1]
            cur = path[i]
            dist = abs(prev[0]-cur[0]) + abs(prev[1]-cur[1]) + abs(prev[2]-cur[2])
            if dist != self.granularity:
                return "Not single hop"

        # Second, check whether it is valid move
        for pos in path:
            if not self.isValidMove(pos[0], pos[1], pos[2]):
                return "Not valid move"

        # Last, check whether it ends up at one of goals
        if not path[-1] in self.__objective:
            return "Last position is not a goal state"

        return "Valid"

    def get_map(self):
        return self.__map
