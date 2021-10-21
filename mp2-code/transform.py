
# transform.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains the transform function that converts the robot arm map
to the maze.
"""
import copy
from arm import Arm
from maze import Maze
from search import *
from geometry import *
from const import *
from util import *

def transformToMaze(arm, goals, obstacles, window, granularity):
    """This function transforms the given 2D map to the maze in MP1.

        Args:
            arm (Arm): arm instance
            goals (list): [(x, y, r)] of goals
            obstacles (list): [(x, y, r)] of obstacles
            window (tuple): (width, height) of the window
            granularity (int): unit of increasing/decreasing degree for angles

        Return:
            Maze: the maze instance generated based on input arguments.

    """
    start_angles = arm.getArmAngle()
    arm_limits = arm.getArmLimit()
    maze_size = []
    for angle_min,angle_max in arm_limits:
        maze_size.append(int((angle_max-angle_min)/granularity + 1))
    arm_num = len(maze_size)
    # pad to 3d
    maze_size += [1]*(3-arm_num)
    start_angles += [0]*(3-arm_num)
    arm_limits += [(0, 0)]*(3-arm_num)
    angle_offsets = [arm_limits[0][0], arm_limits[1][0], arm_limits[2][0]]
    # maze initialized to 3d
    r_maze = [[[SPACE_CHAR for i in range(maze_size[2])] for j in range(maze_size[1])] for k in range(maze_size[0])]
    
    # maze construction
    for a in range(maze_size[0]):
        for b in range(maze_size[1]):
            for r in range(maze_size[2]):
                angles = idxToAngle([a, b, r], angle_offsets, granularity)
                arm.setArmAngle([angles[0], angles[1], angles[2]])
                if not isArmWithinWindow(arm.getArmPos(), window):
                    r_maze[a][b][r] = WALL_CHAR
                else:
                    if doesArmTouchObjects(arm.getArmPosDist(), obstacles):
                        r_maze[a][b][r] = WALL_CHAR
                    else:
                        if doesArmTouchObjects(arm.getArmPosDist(), goals, isGoal=True):
                            if doesArmTipTouchGoals(arm.getEnd(), goals):
                                r_maze[a][b][r] = OBJECTIVE_CHAR
                            else:
                                r_maze[a][b][r] = WALL_CHAR
    #print("size", maze_size)
    start_indices = angleToIdx(start_angles, angle_offsets, granularity)
    r_maze[start_indices[0]][start_indices[1]][start_indices[2]] = START_CHAR
    #print("start", start_indices)
    return Maze(r_maze, angle_offsets, granularity)
