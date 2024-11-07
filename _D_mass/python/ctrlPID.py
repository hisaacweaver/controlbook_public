import numpy as np
import massParam as P

class ctrlPID:
    def __init__(self):
        self.ki = 0.2
        self.kp = 50
        self.kd = 12
        # PD gains
        print('kp: ', self.kp)
        print('kd: ', self.kd)
        # dirty derivative gains
        self.sigma = 0.05  
        #----------------------------------------------------------
        # variables for integrator and differentiator
        self.zdot = P.zdot0  # estimated derivative of theta
        self.z1 = P.z0  # theta delayed by one sample
        self.error_dot = 0.0  # estimated derivative of error
        self.error_d1 = 0.0  # Error delayed by one sample
        self.integrator = 0.0  # integrator

    def update(self, z_r, x):
        z = x[0][0]
        zdot = (2.0*self.sigma - P.Ts) / (2.0*self.sigma + P.Ts) * self.zdot \
            + (2.0 / (2.0*self.sigma + P.Ts)) * ((z - self.z1))
        
        error = z_r - z
        # Anti-windup scheme: only integrate theta when zdot is small
        if abs(self.zdot < 0.08):
            self.integrator = self.integrator \
                + (P.Ts / 2) * (error + self.error_d1)
        # compute the linearized torque using PD control
        F = self.kp * (error) - self.kd * zdot + self.ki * self.integrator
        F = saturate(F, P.F_max)

        self.error_d1 = error
        self.z1 = z
        return F

def saturate(u, limit):
    if abs(u) > limit:
        u = limit * np.sign(u)
    return u