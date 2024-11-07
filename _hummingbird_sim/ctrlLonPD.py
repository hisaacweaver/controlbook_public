import numpy as np
import hummingbirdParam as P

class ctrlLonPD:
    def __init__(self):
        # Design parameters for PD control
        tr = 1.0
        zeta = 0.707

        # Calculate natural frequency
        wn = 2.2 / tr
        b_theta = P.ellT/(P.m1 * P.ell1**2 + P.m2 * P.ell2**2 + P.J1y + P.J2y)

        # PD gains based on the characteristic polynomial
        self.kp = wn**2 / b_theta  # b_theta is a system parameter
        self.kd = 2 * zeta * wn / b_theta
        self.theta_d = 0.0
        print(f"\n\nGains- \nkp:{self.kp} \nkd:{self.kd}\n\n")



    def update(self, x, ref):
        theta = x[1][0]
        thetadot = x[4][0]

        error = ref - theta
                
        torque_equilibrium = 0.0

        force_equilibrium = (P.m1*P.ell1 + P.m2*P.ell2)*P.g / P.ellT * np.cos(theta)
        force_tilde = self.kp * error - self.kd * thetadot
        force = force_equilibrium + force_tilde
        torque = 0.0
        # convert force and torque to pwm signals
        pwm_left = 1/(2*P.km) * (force + torque/P.d)
        pwm_right = 1/(2*P.km) * (force - torque/P.d)
        pwm = np.array([[pwm_left], [pwm_right]])
        pwm = saturate(pwm, 0, 1) 
        return pwm


def saturate(u, low_limit, up_limit):
    if isinstance(u, float) is True:
        u = np.max((np.min((u, up_limit)), low_limit))
    else:
        for i in range(0, u.shape[0]):
            u[i][0] = np.max((np.min((u[i][0], up_limit)), low_limit))
    return u

import lab5_hummingbirdSim




