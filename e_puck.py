from controller import Robot
import numpy as np
from PID_controller import PID
import time
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib import collections as mc

class Epuck(Robot):
    def __init__(self, TIME_STEP):
        super(Epuck, self).__init__()
        self.TIME_STEP = TIME_STEP
        self.axel_length = 0.026
        self.dt = 8.5e-6
        self.positions = []
        
        self.orientation = np.zeros(3)
        self.getDevices()
        self.enable_all()
        
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

    def getDevices(self):
        self.left_sensor = self.getDevice("ps0")
        self.right_sensor = self.getDevice("ps7")
        self.gps_sensor = self.getDevice("gps")
        self.gyro_sensor = self.getDevice("gyro")
        self.left_motor = self.getDevice("left wheel motor")
        self.right_motor = self.getDevice("right wheel motor")
        
    def enable_all(self):
        self.gps_sensor.enable(self.TIME_STEP)
        self.gyro_sensor.enable(self.TIME_STEP)
        self.left_sensor.enable(self.TIME_STEP)
        self.right_sensor.enable(self.TIME_STEP)
    
    def normalize(self,theta):
        return (theta + np.pi)%(2*np.pi) - np.pi
        
    def set_motorVelocity(self, *vel):
        self.left_motor.setVelocity(vel[0])
        self.right_motor.setVelocity(vel[1])
    
    def getPosition(self):
        return self.gps_sensor.getValues()
    
    def getOrientation(self):
        self.orientation += np.array(self.gyro_sensor.getValues())*self.dt
        return self.normalize(self.orientation)
        
    def getMotorVelocity(self, theta_error, theta_dot, x_dot, y_dot):
        
        v_linear = np.sqrt(x_dot**2+y_dot**2)

        if theta_error>0:
            v_r = (theta_error)*2*self.axel_length/0.02
            v_l = 0
        elif theta_error<0:
            v_r = 0
            v_l = -(theta_error)*2*self.axel_length/0.02
        else:
            v_r = 0
            v_l = 0
        
        v_motor = np.array([v_linear+v_l, v_linear+v_r])
        return v_motor.reshape(2,)
        
    def follow_trajectory(self, theta_error):
        
        v_linear = 3

        if theta_error>0:
            v_r = (theta_error)*2*self.axel_length/0.02
            v_l = 0
        elif theta_error<0:
            v_r = 0
            v_l = -(theta_error)*2*self.axel_length/0.02
        else:
            v_r = 0
            v_l = 0
        
        v_l/=3.14
        v_r/=3.14
        
        self.left_motor.setVelocity(v_linear+v_l)
        self.right_motor.setVelocity(v_linear+v_r)
        
        
    def rotmat(self, theta):
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1]])

    def rotmat2theta(self, matrix):
        return np.arctan2(matrix[1,0], matrix[0,0])

    def getThetaError(self, x_error, y_error, theta_now):
        theta_desired = np.arctan2(y_error, x_error)
        
        rotmat_now = self.rotmat(theta_now)
        rotmat_desired = self.rotmat(theta_desired)

        theta_error = self.rotmat2theta(rotmat_desired@np.transpose(rotmat_now))
        return theta_error
    
    def recordPositions(self, *pose):
        return self.positions.append(np.array(pose))
        
    def plotPositions(self, rrt, obstacles):
        poses = np.array(self.positions)
        px = [x for x, y in rrt.G.vertices]
        py = [y for x,y in rrt.G.vertices]
        
        fig, ax = plt.subplots(figsize=(16,9))
        for obs in obstacles:
            square = Rectangle((obs.pos), obs.side_length, obs.side_length)
            ax.add_patch(square)
            
        ax.scatter(px, py, c='cyan')
        ax.scatter(rrt.G.startpos[0], rrt.G.startpos[1], c='black')
        ax.scatter(rrt.G.endpos[0], rrt.G.endpos[1], c='black')

        lines = [(rrt.G.vertices[edge[0]], rrt.G.vertices[edge[1]]) for edge in rrt.G.edges]
        lc = mc.LineCollection(lines, colors='green', linewidths=2)
        ax.add_collection(lc)
        
        if rrt.path is not None:
            paths = [(rrt.path[i], rrt.path[i+1]) for i in range(len(rrt.path)-1)]
            lc2 = mc.LineCollection(paths, colors='black', linewidths=4)
            ax.add_collection(lc2)
        
        ax.plot(poses[:,0], poses[:,1], 'r', linewidth=1)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.autoscale()
        ax.margins(0.2)
        ax.set_title("Robot's Movement vs Generated Trajectory")
        plt.grid()
        plt.show()