import numpy as np
import massParam as P

class ctrlPD:
    def __init__(self):
        self.kp = 4.5
        self.kd = 12
        # PD gains
        print('kp: ', self.kp)
        print('kd: ', self.kd)

    def update(self, z_r, x):
        z = x[0][0]
        zdot = x[1][0]
        # compute the linearized torque using PD control
        F = self.kp * (z_r - z) - self.kd * zdot
        F = saturate(F, P.F_max)
        return F

def saturate(u, limit):
    if abs(u) > limit:
        u = limit * np.sign(u)
    return u