import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.patches import Rectangle
from itertools import product
from utils.grid import Grid_robplan
from utils.car import SimpleCar
from utils.environment import Environment_robplan
from utils.dubins_path import DubinsPath
from utils.astar import Astar
from utils.utils import plot_a_car, get_discretized_thetas, round_theta, same_point

"""
Assumptions:
* Reverse is always allowed
* We're always checking Dubins path
* We're always adding extra costs for changing steering angle, turning and reversing
* Heuristic is always A*
"""


class Node:
    def __init__(self, pos, grid_pos):
        self.pos        = pos
        self.grid_pos   = grid_pos
        self.g          = None
        self.gtemp      = None
        self.f          = None
        self.prev       = None
        self.phi        = 0
        self.m          = None      # 1 if forward, -1 if reverse
        self.branches   = []

    def __eq__(self, other):
        return self.grid_pos == other.grid_pos

    def __hash__(self):
        return hash((self.grid_pos))

class HybridAstar:
    def __init__(self, car, grid, unit_theta=np.pi/2, dt=1e-2):
        self.car        = car
        self.grid       = grid
        self.unit_theta = unit_theta
        self.dt         = dt
        self.thetas     = get_discretized_thetas(self.unit_theta)

        self.start      = self.car.start_pos
        self.goal       = self.car.end_pos

        self.steps      = int(np.sqrt(2) * self.grid.cell_size/self.dt) + 1
        self.arc        = self.steps * self.dt
        self.comb       = list(product([1, -1], [-self.car.max_phi, 0, self.car.max_phi]))      # Combining all possibilities

        self.dubins     = DubinsPath(self.car)
        self.astar      = Astar(self.grid, self.goal[:2])

        """Weights"""
        self.w1         = 0.95      # Weight for A* heuristic
        self.w2         = 0.05      # Weight for Euclidean heuristic
        self.w3         = 0.30      # Extra cost for changing steering angle
        self.w4         = 0.10      # Extra cost for turning
        self.w5         = 2.00      # Extra cost for reversing ??

    def get_node(self, pos):
        """Creates a node for a given position"""
        p           = pos[:2]
        theta       = round_theta(pos[2] % (2*np.pi), self.thetas)

        cell_id     = self.grid.to_cell_id(p)
        grid_pos    = cell_id + [theta]

        return Node(pos, grid_pos)

    def euclidean_heu(self, pos):
        return np.sqrt((self.goal[0] - pos[0])**2 + (self.goal[1] - pos[1])**2)

    def astar_heu(self, pos):
        h1  = self.astar.search_path(pos[:2]) * self.grid.cell_size
        h2  = self.euclidean_heu(pos)

        return (h1 * self.w1) + (h2 * self.w2)

    def get_neighbors(self, node):
        neighbors = []
        for m, phi in self.comb:
            
            """Unwanted actions that we want to skip"""
            if node.m and node.phi == phi and node.m*m == -1:
                # Drive forward, same angle, but now we want to reverse
                # We want to skip this action
                continue

            ### Usikker på om denne kan kommenteres ut eller ei? Vil den ikke bare forhindre at vi reverserer?
            if node.m and node.m == 1 and m == -1:
                # Check if node.m is not None, the node is in "forward" state, and we want to reverse
                # We want to skip this action
                continue

            pos     = node.pos
            branch  = [m, pos[:2]]
            for _ in range(self.steps):
                pos = self.car.step(pos, phi, m)
                branch.append(pos[:2])


            """Checking safety of route"""
            pos1 = node.pos if m == 1 else pos
            pos2 = pos      if m == 1 else node.pos
            if phi == 0:
                # --- We are going straight forward
                safe    = self.dubins.is_straight_route_safe(pos1, pos2)
            else:
                # --- We are turning
                d, c, r = self.car.get_params(pos1, phi)
                safe    = self.dubins.is_turning_route_safe(pos1, pos2, d, c, r)

            if not safe:
                continue

            neighbor        = self.get_node(pos)
            neighbor.phi    = phi
            neighbor.m      = m
            neighbor.prev   = node
            neighbor.g      = node.g + self.arc
            neighbor.gtemp  = node.gtemp + self.arc


            """Extra costs"""
            if phi != node.phi:     # Changing steering angle
                neighbor.g += self.w3 * self.arc

            if phi != 0:            # Turning
                neighbor.g += self.w4 * self.arc

            if m == -1:             # Reversing
                neighbor.g += self.w5 * self.arc


            """Heuristic"""
            neighbor.f = neighbor.g + self.astar_heu(neighbor.pos)


            neighbors.append([neighbor, branch])

        return neighbors

    def best_final_shot(self, open_set, closed_set, best, cost, d_route, n=10):
        """Search best final shot in a given open set"""

        open_set.sort(key=lambda node: node.f)

        for i in range(min(n, len(open_set))):
            best_temp               = open_set[i]
            solutions_temp          = self.dubins.find_tangents(best_temp.pos, self.goal)
            d_temp, c_temp, v_temp  = self.dubins.best_tangent(solutions_temp)

            if v_temp and c_temp + best_temp.gtemp < cost + best.gtemp:
                best    = best_temp
                cost    = c_temp
                d_route = d_temp

        if best in open_set:
            open_set.remove(best)
            closed_set.append(best)

        return best, cost, d_route

    def extract_route(self, node):
        """Backtracking to find the route"""

        route = [(node.pos, node.phi, node.m)]
        while node.prev:
            node = node.prev
            route.append((node.pos, node.phi, node.m))
        
        return list(reversed(route))

    def search_path(self):
        root       = self.get_node(self.start)
        root.g     = float(0)
        root.gtemp = float(0)
        root.f     = root.g + self.astar_heu(root.pos)

        closed_set  = []
        open_set    = [root]

        count = 0
        while open_set:
            count += 1
            best   = min(open_set, key=lambda x: x.f)
            open_set.remove(best)
            closed_set.append(best)

            """Check Dubins path"""
            solutions = self.dubins.find_tangents(best.pos, self.goal)
            d_route, cost, valid = self.dubins.best_tangent(solutions)

            if valid:
                best, cost, d_route = self.best_final_shot(open_set, closed_set, best, cost, d_route)
                route   = self.extract_route(best) + d_route
                path    = self.car.get_path(self.start, route)
                cost   += best.gtemp
                print('Total iteration:', count)

                return path, closed_set

            neighbors = self.get_neighbors(best)
            for neighbor, branch in neighbors:
                
                if neighbor in closed_set:
                    continue

                if neighbor not in open_set:
                    best.branches.append(branch)
                    open_set.append(neighbor)
                elif neighbor.g < open_set[open_set.index(neighbor)].g:
                    best.branches.append(branch)

                    c = open_set[open_set.index(neighbor)]
                    p = c.prev
                    for b in p.branches:
                        if same_point(b[-1], c.pos[:2]):
                            p.branches.remove(b)
                            break

        return None, None

class MapGrid:
    """Defining obstacles for CA4"""
    def __init__(self):
        self.obs = [
            [0, 0, 1, 1],
            [0, 8.2, 4.4, 1.7],
            [2.2, 11.1, 5.7, 6.4],
            [8.4, 12, 3.6, 6],
            [12.5, 12, 3, 6],
            [16, 12, 5.6, 6],            
            [11.15, 4.3, 7.3, 2.8],
            [20.45, 4.3, 7.3, 2.8],
            [25, 0, 15, 2],
            [39, 2, 1, 1],
            [26.4, 14.3, 6, 4],
            [25.7, 21.5, 1, 1],
        ]

def run_HybridAStar(start, goal):
    map     = MapGrid()
    env     = Environment_robplan(map.obs)
    car     = SimpleCar(env, start, goal)
    grid    = Grid_robplan(env)
    hastar  = HybridAstar(car, grid)

    t                   = time.time()
    path, closed_set    = hastar.search_path()
    print('Total time: {}s'.format(round(time.time()-t, 3)))

    if not path:
        print('No valid path')
        return
    
    path_return = []
    for i in range(len(path)):
        path_return.append(path[i].pos)

    """Post-processing to obtain path list"""
    path = path[::5] + [path[-1]]
    branches = []
    bcolors = []
    for node in closed_set:
        for b in node.branches:
            branches.append(b[1:])
            bcolors.append('y' if b[0] == 1 else 'b')

    xl, yl          = [], []
    xl_np1, yl_np1  = [], []
    carl            = []
    dt_s            = int(25)   # Samples for Gazebo

    for i in range(len(path)):
        xl.append(path[i].pos[0])
        yl.append(path[i].pos[1])
        carl.append(path[i].model[0])
        if i==0 or i==len(path):
            xl_np1.append(path[i].pos[0])
            yl_np1.append(path[i].pos[1])            
        elif dt_s*i<len(path):
            xl_np1.append(path[i*dt_s].pos[0])
            yl_np1.append(path[i*dt_s].pos[1])

    """ Defining waypoints """
    xl_np   = np.array(xl_np1)
    xl_np   = xl_np - 10
    yl_np   = np.array(yl_np1)
    yl_np   = yl_np - 10
    global WAYPOINTS
    WAYPOINTS = np.column_stack([xl_np, yl_np])

    start_state = car.get_car_state(car.start_pos)
    end_state = car.get_car_state(car.end_pos)

    """Animation - Copied this from Ass1"""
    # plot and annimation
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(0, env.lx)
    ax.set_ylim(0, env.ly)
    ax.set_aspect("equal")

    """Set grid on"""
    ax.set_xticks(np.arange(0, env.lx, grid.cell_size))
    ax.set_yticks(np.arange(0, env.ly, grid.cell_size))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    plt.grid(which='both')
    #ax.set_xticks([])
    #ax.set_yticks([])
    
    for ob in env.obs:
        ax.add_patch(Rectangle((ob.x, ob.y), ob.w, ob.h, fc='gray', ec='k'))
    
    ax.plot(car.start_pos[0], car.start_pos[1], 'ro', markersize=6)
    ax = plot_a_car(ax, end_state.model)
    ax = plot_a_car(ax, start_state.model)

    _branches = LineCollection([], linewidth=1)
    ax.add_collection(_branches)

    _path, = ax.plot([], [], color='lime', linewidth=2)
    _carl = PatchCollection([])
    ax.add_collection(_carl)
    _path1, = ax.plot([], [], color='w', linewidth=2)
    _car = PatchCollection([])
    ax.add_collection(_car)
    
    frames = len(branches) + len(path) + 1

    def init():
        _branches.set_paths([])
        _path.set_data([], [])
        _carl.set_paths([])
        _path1.set_data([], [])
        _car.set_paths([])

        return _branches, _path, _carl, _path1, _car

    def animate(i):

        edgecolor = ['k']*5 + ['r']
        facecolor = ['y'] + ['k']*4 + ['r']

        if i < len(branches):
            _branches.set_paths(branches[:i+1])
            _branches.set_color(bcolors)
        
        else:
            _branches.set_paths(branches)

            j = i - len(branches)

            _path.set_data(xl[min(j, len(path)-1):], yl[min(j, len(path)-1):])

            sub_carl = carl[:min(j+1, len(path))]
            _carl.set_paths(sub_carl[::4])
            _carl.set_edgecolor('k')
            _carl.set_facecolor('m')
            _carl.set_alpha(0.1)
            _carl.set_zorder(3)

            _path1.set_data(xl[:min(j+1, len(path))], yl[:min(j+1, len(path))])
            _path1.set_zorder(3)

            _car.set_paths(path[min(j, len(path)-1)].model)
            _car.set_edgecolor(edgecolor)
            _car.set_facecolor(facecolor)
            _car.set_zorder(3)

        return _branches, _path, _carl, _path1, _car

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=frames,
                                  interval=1, repeat=True, blit=True)

    plt.show()

    return path_return
    

if __name__ == '__main__':
    
    """Parameters"""
    safety_distance = 0.76
    WP0 = [0.5, 1.0 + safety_distance, -np.pi/2]
    WP1 = [14.8, 4.3 - safety_distance, np.pi/2]
    WP2 = [24.1, 7.1 + safety_distance, -np.pi/2]
    WP3 = [26.2, 21.5 - safety_distance, np.pi/2]
    WP4 = [39.5, 3.0 + safety_distance, -np.pi/2]
    WP5 = [7.0 , 19.5, 0]
    WP6 = [29.4, 14.3 - safety_distance, np.pi/2]

    start   = WP4
    goal    = WP1

    print('----- Executing hybrid A* -----')
    path = run_HybridAStar(start, goal)

    ### Finding track with fewer points
    prev = [0, 0, 0]
    track = []
    for p in path:
        if abs(p[2] - prev[2]) > 0.1: #or abs(p[0] - prev[0]) > 1 or abs(p[1] - prev[1]) > 1:
            if abs(p[0] - prev[0]) > 0.1 or abs(p[1] - prev[1]) > 0.1:
                track.append(p)
                prev = p

    ### Shifting waypoints
    shift = []
    for point in track:
        shift.append([point[0] - 20, point[1] - 11.25, point[2]])

    print('Length: ', len(path), len(track))