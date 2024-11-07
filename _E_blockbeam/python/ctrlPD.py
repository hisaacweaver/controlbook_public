import numpy as np
import blockbeamParam as P

class ctrlPD:
    def __init__(self):
        ####################################################
        #       PD Control: Time Design Strategy
        ####################################################
        # tuning parameters
        tr_z = 1          # Rise time for inner loop (theta)
        zeta_th = 0.707       # inner loop Damping Coefficient

        # saturation limits
        F_max = 5             		  # Max Force, N
        error_max = 1        		  # Max step size,m
        theta_max = 30.0 * np.pi / 180.0  # Max theta, rads

        #---------------------------------------------------
        #                    Inner Loop
        #---------------------------------------------------
        # parameters of the open loop transfer function
        ze = P.length/2.0
        beta = P.length / (P.m2*P.length**2/3.0 + P.m1*ze**2)
        M = 10         # Time scale separation 
        tr_th = tr_z / M

        # coefficients for desired inner loop
        wn_th = 2.2 / tr_th     # Natural frequency


        # compute gains
        self.kp_th = wn_th**2/beta
        self.kd_th = (2.0 * zeta_th * wn_th) / beta
        DC_gain = 1.0
        #---------------------------------------------------
        #                    Outer Loop
        #---------------------------------------------------
        # coefficients for desired outer loop
        zeta_z = 0.707     # outer loop Damping Coefficient
        wn_z = 2.2 / tr_z  # desired natural frequency

        des_poles = np.array([zeta_th*wn_th, wn_th**2, zeta_z*wn_z, wn_z**2])
        print(des_poles)

        # compute gains
        
        self.kd_z = 2*wn_z*zeta_z / -P.g
        self.kp_z = -wn_z**2 / P.g 


        # print control gains to terminal        
        print('DC_gain', DC_gain)
        print('kp_th: ', self.kp_th)
        print('kd_th: ', self.kd_th)
        print('kp_z: ', self.kp_z)
        print('kd_z: ', self.kd_z)


    def update(self, z_r, state):
        z = state[0][0]
        theta = state[1][0]
        zdot = state[2][0]
        thetadot = state[3][0]

        # the reference angle for theta comes from the
        # outer loop PD control
        theta_r = self.kp_z * (z_r - z) - self.kd_z * zdot

        F_tilde =  self.kp_th * (theta_r - theta) - self.kd_th * thetadot

        F_fl = P.m1 * P.g * (z / P.length) + P.m2 * P.g / 2.0 

        F_unsat = F_tilde + F_fl
        F = saturate(F_unsat, P.F_max) 
        return F

def saturate(u, limit):
    if abs(u) < limit:
        u = limit * np.sign(u)
    return u


if __name__ == "__main__":
    import hw07_blockbeamSim


