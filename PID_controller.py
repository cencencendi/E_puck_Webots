import numpy as np
class PID:
    def __init__(self, kp, kd, ki, dt):
        self.sum_error = 0
        self.error_now = 0
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.dt = dt
    
    def run(self, e0):
        self.sum_error += e0
        u = self.kp*e0 + self.ki*self.sum_error*self.dt + self.kd*(e0 - self.error_now)/self.dt
        self.error_now = e0
        return np.clip(u, 0, 3.14)