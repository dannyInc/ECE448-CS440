from copy import deepcopy
import sys

def MinKeyExtract(key, visited):
    min = sys.maxsize
    min_vertex = None
    for v in visited.keys():
        if visited[v] == False and key[v] < min:
            min = key[v]
            min_vertex = v
    return min_vertex

def manhattan(node_a, node_b):
    if not isinstance(node_a, tuple):
        return "position must be tuple"
    if not isinstance(node_b, tuple):
        return "position must be tuple"
    return abs(node_a[0]-node_b[0]) + abs(node_a[1]-node_b[1])

def get_MST(goals, edge_weights, start_goal):
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
        
    key[start_goal] = 0

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

if __name__ == "__main__":

    objs_left = [(10,8),(5,6),(1,4),(4,4),(9,19),(6,13)]
    edge_weights = {}
    for i in objs_left:
        for j in objs_left:
                edge_weights[(i, j)] = manhattan(i, j)
    
    for o in objs_left:
        print(get_MST(objs_left, edge_weights, o))
