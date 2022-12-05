import numpy as np
from PID_controller import PID
from e_puck import Epuck
from rrt import RRT, Obstacle
import time

TIME_STEP = 64

epuck = Epuck(TIME_STEP)

#==================================PID TUNING============================== 
kp_theta, kd_theta, ki_theta = 0,0,0            
kp_xy, kd_xy, ki_xy = 10, 0.1, 0.5

pid_theta = PID(kp = kp_theta, kd = kd_theta, ki = ki_theta, dt = epuck.dt)
pid_x = PID(kp = kp_xy, kd = kd_xy, ki = ki_xy, dt = 64e-3)
pid_y = PID(kp = kp_xy, kd = kd_xy, ki = ki_xy, dt = 64e-3)
#==========================================================================

#=============================Generate Trajectory==========================
startpos = (0., 0.)
endpos = (2.0, 2.0)
obstacles = [Obstacle(pos, side_length = 0.5) for pos in[[0.625, 0.375],[0.875,1.125]]]
n_iter = 1000
stepSize = 0.3

start_time = time.time()
print(f"Generating trajectories using RRT Algorithm from {startpos} to {endpos}...")
rrtku = RRT(startpos, endpos, obstacles, n_iter, stepSize)
trajectories = rrtku.find_path()
length = trajectories.shape[0] 
trajectories = np.delete(trajectories, obj = 0, axis = 0)
print(f"Trajectories generated successfully with duration: {time.time()-start_time}s")
print(f"Trajectories:\n{trajectories}")
print(f"Number of waypoints: {length}")
#==========================================================================
now = 0
while epuck.step(TIME_STEP) != -1:
    target_position = trajectories[now,:]
    x_now, y_now, _ = epuck.getPosition()
    _, _, zAngle = epuck.getOrientation()

    x_error = target_position[0] - x_now
    y_error = target_position[1] - y_now
    
    theta_error = epuck.getThetaError(x_error, y_error, zAngle)
    epuck.recordPositions(x_now, y_now)
    if now<length-2:
        epuck.follow_trajectory(theta_error)
            
    elif now>=length-2:
        lw, rw = epuck.getMotorVelocity(theta_error,  
                                    pid_theta.run(theta_error),
                                    pid_x.run(x_error),
                                    pid_y.run(y_error))
                                    
        epuck.left_motor.setVelocity(lw)
        epuck.right_motor.setVelocity(rw)
        
    if abs(x_error)<1e-1 and abs(y_error)<1e-1 and now<length-2:
        print(f"Robot's position to waypoint-{now+1} error: {np.sqrt(x_error**2+y_error**2):.5f}")
        now+=1
    elif abs(x_error)<1e-3 and abs(y_error)<1e-3 and now>=length-2:
        epuck.left_motor.setVelocity(0)
        epuck.right_motor.setVelocity(0)
        print("="*52)
        print(f"|| Robot's position now at:\t\t\t\t||")
        print(f"\n|| x: {x_now:.4f}\t\t\t\t\t\t||")
        print(f"\n|| y: {y_now:.4f}\t\t\t\t\t\t||")
        print(f"\n|| Position Error: {np.sqrt(x_error**2+y_error**2):.4f}\t\t\t\t||")
        print("="*52)
        epuck.plotPositions(rrtku, obstacles)
        break
            
