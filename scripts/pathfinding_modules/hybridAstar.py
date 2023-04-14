import numpy as np
import math

class Node:
    def __init__(self, x, y, theta, g, h, parent=None):
        self.x = x
        self.y = y
        self.theta = theta
        self.g = g
        self.h = h
        self.parent = parent

class HybridAStar:
    def __init__(self, obstacles, resolution):
        self.obstacles = obstacles
        self.resolution = resolution
        self.motion = self.generate_motion()

    def search(self, start, goal):
        start = Node(round(start[0]/self.resolution), round(start[1]/self.resolution), round(start[2]), 0.0, self.heuristic(start, goal))
        goal = Node(round(goal[0]/self.resolution), round(goal[1]/self.resolution), round(goal[2]), 0.0, 0.0)
        closed_set = []
        open_set = [start]
        while open_set:
            current = min(open_set, key=lambda o:o.g + o.h)
            if current.x == goal.x and current.y == goal.y and current.theta == goal.theta:
                return self.extract_path(current)
            open_set.remove(current)
            closed_set.append(current)
            for move in self.motion:
                next_x = current.x + move[0]
                next_y = current.y + move[1]
                next_theta = current.theta + move[2]
                if next_theta >= 360:
                    next_theta -= 360
                if next_theta < 0:
                    next_theta += 360
                if next_x < 0 or next_x >= self.obstacles.shape[0] or next_y < 0 or next_y >= self.obstacles.shape[1] or self.obstacles[next_x][next_y]:
                    continue
                next_node = Node(next_x, next_y, next_theta, current.g + move[3], self.heuristic((next_x, next_y, next_theta), goal), current)
                if any(next_node.x == node.x and next_node.y == node.y and next_node.theta == node.theta for node in closed_set):
                    continue
                open_set_nodes = [node for node in open_set if next_node.x == node.x and next_node.y == node.y and next_node.theta == node.theta]
                if not open_set_nodes:
                    open_set.append(next_node)
                else:
                    open_node = open_set_nodes[0]
                    if next_node.g < open_node.g:
                        open_node.g = next_node.g
                        open_node.parent = current

    def extract_path(self, node):
        path = [(node.x*self.resolution, node.y*self.resolution, node.theta)]
        while node.parent:
            node = node.parent
            path.append((node.x*self.resolution, node.y*self.resolution, node.theta))
        return path[::-1]

    def generate_motion(self):
        motions = []
        for steer in range(-30, 31, 5):
            for d in range(1, 4):
                motions.append((int(round(math.sin(math.radians(steer))*d/self.resolution)), int(round(math.cos(math.radians(steer))*d/self.resolution)), steer, d))
        return motions

    def heuristic(self, start, goal):
        return math.sqrt((start[0] - goal[0])**2 + (start[1] - goal[1])**2)                     

if __name__ == '__main__':
    # Define the map and the resolution
    obstacles = np.zeros((50, 50))
    obstacles[20:30, 10:20] = 1
    obstacles[40:50, 30:40] = 1
    resolution = 0.1

    # Create an instance of HybridAStar class
    hybrid_astar = HybridAStar(obstacles, resolution)

    # Define the starting and goal positions
    start = (5, 5, 0)
    goal = (45, 45, 0)

    # Find the path using Hybrid A* algorithm
    path = hybrid_astar.search(start, goal)

    # Print the path
    print(path)