#!/usr/bin/env python3
import rospy
import os
import tf
import numpy as np
import matplotlib.pyplot as plt
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from math import pi, sqrt, atan2, tan
from os import system, name
import time
import re
import fileinput
import sys
import argparse
import random
import matplotlib.animation as animation
from datetime import datetime
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.patches import Rectangle
from itertools import product
from utils.grid import Grid_robplan
from utils.car import SimpleCar
from utils.environment import Environment_robplan
from utils.dubins_path import DubinsPath
from utils.astar import Astar
from utils.utils import plot_a_car, get_discretized_thetas, round_theta, same_point
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import shutil
import copy
import pyplanning as pp

""" ----------------------------------------------------------------------------------
Mission planner for Autonomos robots: TTK4192, NTNU
Date: 24.04.23
Characteristics: AI planning, GNC, hybrid A*, ROS
Robot: Turtlebot3
Version: 1.1
""" 


# 1) Program here your AI planner 
"""
Graph plan ---------------------------------------------------------------------------
"""

class PlanningStep():
    def __init__(self, action, objects):
        self.action = action
        self.objects = objects

    def get_waypoints(self):
        if self.action == "move":
            WP_from = self.objects[0].split("WP")[1]
            WP_to   = self.objects[1].split("WP")[1]
            return WP_from, WP_to
        else:
            WP      = self.objects[0].split("WP")[1]
            return WP


#2) GNC module (path-followig and PID controller for the robot)
"""  Robot GNC module ----------------------------------------------------------------------
"""
class PID:
    """
    Discrete PID control
    """
    def __init__(self, P=0.0, I=0.0, D=0.0, Derivator=0, Integrator=0, Integrator_max=10, Integrator_min=-10):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.Derivator = Derivator
        self.Integrator = Integrator
        self.Integrator_max = Integrator_max
        self.Integrator_min = Integrator_min
        self.set_point = 0.0
        self.error = 0.0

    def update(self, current_value):
        PI = 3.1415926535897
        self.error = self.set_point - current_value
        if self.error > pi:  # specific design for circular situation
            self.error = self.error - 2*pi
        elif self.error < -pi:
            self.error = self.error + 2*pi
        self.P_value = self.Kp * self.error
        self.D_value = self.Kd * ( self.error - self.Derivator)
        self.Derivator = self.error
        self.Integrator = self.Integrator + self.error
        if self.Integrator > self.Integrator_max:
            self.Integrator = self.Integrator_max
        elif self.Integrator < self.Integrator_min:
            self.Integrator = self.Integrator_min
        self.I_value = self.Integrator * self.Ki
        PID = self.P_value + self.I_value + self.D_value
        return PID

    def setPoint(self, set_point):
        self.set_point = set_point
        self.Derivator = 0
        self.Integrator = 0

    def setPID(self, set_P=0.0, set_I=0.0, set_D=0.0):
        self.Kp = set_P
        self.Ki = set_I
        self.Kd = set_D

class turtlebot_move():
    """
    Path-following module
    """
    def __init__(self):
        rospy.init_node('turtlebot_move', anonymous=False)
        rospy.loginfo("Press CTRL + C to terminate")
        rospy.on_shutdown(self.stop)

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.pid_theta = PID(0,0,0)  # initialization

        self.odom_sub = rospy.Subscriber("odom", Odometry, self.odom_callback) # subscribing to the odometer
        self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)        # reading vehicle speed
        self.vel = Twist()
        self.rate = rospy.Rate(10)
        self.counter = 0
        self.trajectory = list()

        # track a sequence of waypoints
        for point in WAYPOINTS:
            self.move_to_point(point[0], point[1])
            rospy.sleep(1)
        self.stop()
        rospy.logwarn("Action done.")

        # plot trajectory
        data = np.array(self.trajectory)
        np.savetxt('trajectory.csv', data, fmt='%f', delimiter=',')
        plt.plot(data[:,0],data[:,1])
        plt.show()


    def move_to_point(self, x, y):
        # Here must be improved the path-following ---
        # Compute orientation for angular vel and direction vector for linear velocity

        diff_x = x - self.x
        diff_y = y - self.y
        direction_vector = np.array([diff_x, diff_y])
        direction_vector = direction_vector/sqrt(diff_x*diff_x + diff_y*diff_y)  # normalization
        theta = atan2(diff_y, diff_x)

        # We should adopt different parameters for different kinds of movement
        self.pid_theta.setPID(1, 0, 0)     # P control while steering
        self.pid_theta.setPoint(theta)
        rospy.logwarn("### PID: set target theta = " + str(theta) + " ###")

        
        # Adjust orientation first
        while not rospy.is_shutdown():
            angular = self.pid_theta.update(self.theta)
            if abs(angular) > 0.2:
                angular = angular/abs(angular)*0.2
            if abs(angular) < 0.01:
                break
            self.vel.linear.x = 0
            self.vel.angular.z = angular
            self.vel_pub.publish(self.vel)
            self.rate.sleep()

        # Have a rest
        self.stop()
        self.pid_theta.setPoint(theta)
        self.pid_theta.setPID(1, 0.02, 0.2)  # PID control while moving

        # Move to the target point
        while not rospy.is_shutdown():
            diff_x = x - self.x
            diff_y = y - self.y
            vector = np.array([diff_x, diff_y])
            linear = np.dot(vector, direction_vector) # projection
            if abs(linear) > 0.2:
                linear = linear/abs(linear)*0.2

            angular = self.pid_theta.update(self.theta)
            if abs(angular) > 0.2:
                angular = angular/abs(angular)*0.2

            if abs(linear) < 0.01 and abs(angular) < 0.01:
                break
            self.vel.linear.x = 1.5*linear   # Here can adjust speed
            self.vel.angular.z = angular
            self.vel_pub.publish(self.vel)
            self.rate.sleep()
        self.stop()
    def stop(self):
        self.vel.linear.x = 0
        self.vel.angular.z = 0
        self.vel_pub.publish(self.vel)
        rospy.sleep(1)

    def odom_callback(self, msg):
        # Get (x, y, theta) specification from odometry topic
        quarternion = [msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,\
                    msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(quarternion)
        self.theta = yaw
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        # Make messages saved and prompted in 5Hz rather than 100Hz
        self.counter += 1
        if self.counter == 20:
            self.counter = 0
            self.trajectory.append([self.x,self.y])
            #rospy.loginfo("odom: x=" + str(self.x) + ";  y=" + str(self.y) + ";  theta=" + str(self.theta))



# 3) Program here your path-finding algorithm --> Done 
""" Hybrid A-star pathfinding --------------------------------------------------------------------
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
    
    #for i in range(len(path)):
    #    print(path[i].pos)

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
    

#4) Program here the turtlebot actions (based on your AI planner)
"""
Turtlebot 3 actions-------------------------------------------------------------------------
"""

class TakePhoto:
    def __init__(self):

        self.bridge = CvBridge()
        self.image_received = False

        # Connect image topic
        img_topic = "/camera/rgb/image_raw"
        self.image_sub = rospy.Subscriber(img_topic, Image, self.callback)

        # Allow up to one second to connection
        rospy.sleep(1)

    def callback(self, data):

        # Convert image to OpenCV format
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        self.image_received = True
        self.image = cv_image

    def take_picture(self, img_title):
        if self.image_received:
            # Save an image
            cv2.imwrite(img_title, self.image)
            return True
        else:
            return False
        
def taking_photo():
    # Initialize
    camera = TakePhoto()

    # Default value is 'photo.jpg'
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")
    img_title = rospy.get_param('~image_title', 'photo'+dt_string+'.jpg')

    if camera.take_picture(img_title):
        rospy.loginfo("Saved image " + img_title)
    else:
        rospy.loginfo("No images received")
	#eog photo.jpg
    # Sleep to give the last log messages time to be sent

	# saving photo in a desired directory
    file_source = '/home/catkin_ws/'
    file_destination = '/home/catkin_ws/src/ca4_ttk4192/scripts'
    g='photo'+dt_string+'.jpg'

    shutil.move(file_source + g, file_destination)
    rospy.sleep(1)

def move_robot(WPx, WPy):
    # QUESTION: Is it correct to run hybrid A*?
    # How can turtlebot_move() find the correcponding waypoints?
    # Are the waypoints indexed in bottom left corner or origin?

    print("Moving robot from WP{} to WP{}".format(WPx, WPy))
    print("Computing hybrid A* path")

    # Waypoints defined with origin in bottom left corner for hybrid A*
    safety_distance = 0.76
    WP = [[0.5, 1.0 + safety_distance, -np.pi/2],
         [14.8, 4.3 - safety_distance, np.pi/2],
         [24.1, 7.1 + safety_distance, -np.pi/2],
         [26.2, 21.5 - safety_distance, np.pi/2],
         [39.5, 3.0 + safety_distance, -np.pi/2],
         [7.0 , 19.5, 0],
         [29.4, 14.3 - safety_distance, np.pi/2]]
    
    start_pos = WP[int(WPx)]
    end_pos = WP[int(WPy)]

    run_HybridAStar(start_pos, end_pos)
    print("Executing path following")
    turtlebot_move()

def take_picture(WPx):
    # QUESTION: Is this the correct way of doing this?

    print("Taking IR picture at WP{} ...".format(WPx))
    taking_photo()
    time.sleep(5)

def inspect_valve(WPx):
    # QUESTION: Correct? Robot not supposed to do anything?

    print("Inspecting valve at WP{} ...".format(WPx))
    time.sleep(5)

def charge_battery(WPx):
    # QUESTION: Correct?

    print("Charging battery at WP{}".format(WPx))
    time.sleep(5)

def Manipulate_OpenManipulator_x():
    # QUESTION: Is this for bonus task? How to start?

    print("Executing manipulate a weight")
    time.sleep(5)

def making_turn_exe():
    print("Executing Make a turn")
    time.sleep(1)
    #Starts a new node
    #rospy.init_node('turtlebot_move', anonymous=True)
    velocity_publisher = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    vel_msg = Twist()

    # Receiveing the user's input
    print("Let's rotate your robot")
    #speed = input("Input your speed (degrees/sec):")
    #angle = input("Type your distance (degrees):")
    #clockwise = input("Clockwise?: ") #True or false

    speed = 5
    angle = 180
    clockwise = True

    #Converting from angles to radians
    angular_speed = speed*2*pi/360
    relative_angle = angle*2*pi/360

    #We wont use linear components
    vel_msg.linear.x=0
    vel_msg.linear.y=0
    vel_msg.linear.z=0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0

    # Checking if our movement is CW or CCW
    if clockwise:
        vel_msg.angular.z = -abs(angular_speed)
    else:
        vel_msg.angular.z = abs(angular_speed)
    # Setting the current time for distance calculus
    t0 = rospy.Time.now().to_sec()
    current_angle = 0   #should be from the odometer

    while(current_angle < relative_angle):
        velocity_publisher.publish(vel_msg)
        t1 = rospy.Time.now().to_sec()
        current_angle = angular_speed*(t1-t0)

    #Forcing our robot to stop
    vel_msg.angular.z = 0
    velocity_publisher.publish(vel_msg)
    #rospy.spin()

# QUESTION: These not used?
"""
def check_pump_picture_ir(WPx):
    a=0
    while a<3:
        print("Taking IR picture at WP{} ...".format(WPx))
        time.sleep(1)
        a=a+1
    time.sleep(5)

def check_seals_valve_picture_eo(WPx):
    a=0
    while a<3:
        print("Taking EO picture at WP{} ...".format(WPx))
        time.sleep(1)
        a=a+1
    time.sleep(5)
"""


# Define the global varible: WAYPOINTS  Wpts=[[x_i,y_i]];
global WAYPOINTS
safety_distance = 0.76
# QUESTION: Are these correct? Correct way of defining them?
WAYPOINTS = [[-19.5,-10.25 + safety_distance ],   #[0.5, 1.0 + safety_distance],
             [-5.20, -6.95 - safety_distance ],   #[14.8, 4.3 - safety_distance],
             [ 4.10, -4.15 + safety_distance ],   #[24.1, 7.1 + safety_distance],
             [ 6.20, 10.25 - safety_distance ],   #[26.2, 21.5 - safety_distance],
             [ 19.5, -8.25 + safety_distance ],   #[39.5, 3.0 + safety_distance],
             [-13.0, 8.25 ],                      #[7.0 , 19.5],
             [ 9.40, 3.05 - safety_distance ]]    #[29.4, 14.3 - safety_distance]]


# 5) Program here the main commands of your mission planner code
""" Main code ---------------------------------------------------------------------------
"""
if __name__ == '__main__':
    try:
        print()
        print("**************** TTK4192 - Assignment 4 *********************")
        print()
        print("AI planners: GraphPlan")
        print("Path-finding: Hybrid A*")
        print("GNC Controller: PID path-following")
        print("Robot: Turtlebot3 waffle-pi")
        print("Date: 24.04.23")
        print()
        print("**************************************************************")
        print()
        print("Press Intro to start ...")
        input_t = input("")
        # 5.0) Testing the GNC module (uncomment lines to test)
        #print("testing GNC module, move between way-points")
        #WAYPOINTS = [[-19.5,-9.25],[-5.2, -6.95 - 0.76]]
        #turtlebot_move()

		# 5.1) Starting the AI Planner
        domain_file = "/home/marie/catkin_ws/src/ca4_ttk4192/scripts/ai_planner_modules/PDDL_domain/domain.pddl"
        problem_file = "/home/marie/catkin_ws/src/ca4_ttk4192/scripts/ai_planner_modules/PDDL_domain/problem.pddl"

        domain, problem = pp.load_pddl(domain_file, problem_file)
        print('----- Finding plan -----')
        plan = pp.solvers.graph_plan(problem, 1000, False)

        if plan is not None:
            print("Plan found:")
            print(plan, "\n")
        else:
            print("Planning failed.")


    
        # 5.2) Reading the plan 
        print("  ")
        print("----- Reading the plan -----")

        plan_general = []
        for i in range(len(plan)):
            lst = list(plan[i+1])[0]
            action = lst.action.name
            obj = []
            for o in lst.objects:
                obj.append(str(o))
            plan_general.append(PlanningStep(action, obj))

        

        # 5.3) Start mission execution 
        print("   ")
        print("----- Starting mission execution -----")

        battery         = 100
        task_finished   = 0
        task_total      = len(plan_general)
        i_ini           = 0

        while i_ini < task_total:

            plan_temp = plan_general[i_ini]
            print('   ')
            print('Step {}: {}, {}'.format(i_ini+1, plan_temp.action, plan_temp.objects))

            if plan_temp.action == "move":
                WPx, WPy = plan_temp.get_waypoints()
                move_robot(WPx, WPy)
                time.sleep(1)

            if plan_temp.action == "take-picture":
                print("Taking picture at WP{}".format(plan_temp.get_waypoints()))
                #taking_photo() <- incorporated into function above
                time.sleep(1)

            if plan_temp.action == "inspect-valve":
                WP = plan_temp.get_waypoints()
                inspect_valve(WP)
                #check_seals_valve_picture_eo(WP)
                time.sleep(1)

            if plan_temp.action == "charge-robot":
                WP = plan_temp.get_waypoints()
                charge_battery(WP)
                time.sleep(1)

            i_ini = i_ini+1  # Next tasks


        print("")
        print("--------------------------------------")
        print("All tasks were performed successfully")
        time.sleep(10)  

    except rospy.ROSInterruptException:
        rospy.loginfo("Action terminated.")