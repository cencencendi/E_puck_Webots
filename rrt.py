import numpy as np
from random import random
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import collections as mc

class RRT:
    def __init__(self, startpos, endpos, obstacles, n_iter, stepSize):
        self.startpos = startpos
        self.endpos = endpos
        self.obstacles = obstacles
        self.n_iter = n_iter
        self.stepSize = stepSize
        self.G = Graph(startpos, endpos)
        self.path = None
    
    def find_path(self):
        for i in range(self.n_iter):
            randvex = self.G.randomPosition()
            if isInObstacle(randvex, self.obstacles):
                continue
            nearvex, nearidx = nearest(self.G, randvex, self.obstacles)
            if nearvex is None:
                continue

            newvex = newvertex(randvex, nearvex, self.stepSize)

            newidx = self.G.add_vex(newvex)
            dist = distance(newvex, nearvex)
            self.G.add_edge(newidx, nearidx, dist)

            dist = distance(newvex, self.G.endpos)
            if dist < 0.5:
                endidx = self.G.add_vex(self.G.endpos)
                self.G.add_edge(newidx, endidx, dist)
                self.G.success = True
                break
            
        if self.G.success == True:
            self.path = self.dijkstra()
        
        return self.path


    def dijkstra(self):

        srcIdx = self.G.vex2idx[self.G.startpos]
        dstIdx = self.G.vex2idx[self.G.endpos]

        # build dijkstra
        nodes = list(self.G.neighbors.keys())
        dist = {node: float('inf') for node in nodes}
        prev = {node: None for node in nodes}
        dist[srcIdx] = 0

        while nodes:
            curNode = min(nodes, key=lambda node: dist[node])
            nodes.remove(curNode)
            if dist[curNode] == float('inf'):
                break

            for neighbor, cost in self.G.neighbors[curNode]:
                newCost = dist[curNode] + cost
                if newCost < dist[neighbor]:
                    dist[neighbor] = newCost
                    prev[neighbor] = curNode

        # retrieve path
        path = deque()
        curNode = dstIdx
        while prev[curNode] is not None:
            path.appendleft(np.array(self.G.vertices[curNode]))
            curNode = prev[curNode]
        path.appendleft(np.array(self.G.vertices[curNode]))
        return np.array(path)

class Line:
    def __init__(self, pos1, pos2):
        self.pos1 = pos1
        self.pos2 = pos2
        self.m = (pos2[1]-pos1[1])/(pos2[0]-pos1[0])
        
    def makepoints(self, n_points):
        self.points = []
        for i in np.linspace(self.pos1[0], self.pos2[0], n_points+1):
            point = self.m*(i-self.pos1[0])+self.pos1[1]
            self.points.append([i,point])
        return self.points 

class Obstacle:
    def __init__(self, pos, side_length):
        self.pos = pos
        self.side_length = side_length
        self.edges = np.array([[pos[0]-side_length, side_length+pos[0]],
                               [pos[1]-side_length, side_length+pos[1]]])

class Graph:
    def __init__(self, startpos, endpos):
        self.startpos = startpos
        self.endpos = endpos

        self.vertices = [startpos]
        self.edges = []
        self.success = False

        self.vex2idx = {startpos:0}
        self.neighbors = {0:[]}
        self.distances = {0:0.}

        self.sx = endpos[0] - startpos[0]
        self.sy = endpos[1] - startpos[1]

    def add_vex(self, pos):
        try:
            idx = self.vex2idx[pos]
        except:
            idx = len(self.vertices)
            self.vertices.append(pos)
            self.vex2idx[pos] = idx
            self.neighbors[idx] = []
        return idx

    def add_edge(self, idx1, idx2, cost):
        self.edges.append((idx1, idx2))
        self.neighbors[idx1].append((idx2, cost))
        self.neighbors[idx2].append((idx1, cost))

    def randomPosition(self):
        rx = random()
        ry = random()

        posx = self.startpos[0] - (self.sx / 2.) + rx * self.sx * 2
        posy = self.startpos[1] - (self.sy / 2.) + ry * self.sy * 2
        return posx, posy


def distance(pos1, pos2):
    return np.sqrt(np.sum([(x-y)**2 for x,y in zip(pos1,pos2)]))

def isInObstacle(position, obstacles):
    for obstacle in obstacles:
        if (position[0]>obstacle.edges[0,0] and position[0]<obstacle.edges[0,1]) and (position[1]>obstacle.edges[1,0] and position[1]<obstacle.edges[1,1]):
            return True
    return False

def isIntersect(line, obstacles):
    if np.any([isInObstacle(point, obstacles) for point in line.makepoints(100)]):
        return True
    return False

def nearest(G, vex, obstacles):
    Nvex = None
    Nidx = None
    minDist = float('inf')

    for idx, v in enumerate(G.vertices):
        line = Line(v, vex)
        if isIntersect(line, obstacles):
            continue
        dist = distance(v, vex)
        if dist < minDist:
            minDist = dist
            Nidx = idx
            Nvex = v

    return Nvex, Nidx

def newvertex(randvex, nearvex, stepSize):
    dirn = np.array(randvex) - np.array(nearvex)
    length = distance(randvex, nearvex)
    dirn = (dirn/length)*min(stepSize, length)

    return (nearvex[0]+dirn[0], nearvex[1]+dirn[1])


    