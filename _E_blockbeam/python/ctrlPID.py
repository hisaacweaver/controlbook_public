import numpy as np
import blockbeamParam as P

class ctrlPID:
    def __init__(self):
        ####################################################
        #       PD Control: Time Design Strategy
        ####################################################
        # tuning parameters
        tr_z = 3          # Rise time for inner loop (theta)
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

        # compute gains

        # dirty derivative gain
        self.sigma = 0.05  # cutoff freq for dirty derivative
        self.beta = (2 * self.sigma - P.Ts) \
            / (2 * self.sigma + P.Ts) 
        
        self.kd_z = 2*wn_z*zeta_z / -P.g
        self.kp_z = -wn_z**2 / P.g 

        # self.kp_th = 31.370370370370374
        # self.kd_th = 6.048777777777778
        # self.kp_z = -0.054875283446712025
        # self.kd_z = -0.1058095238095238
        self.ki_z = -0.05
        # self.ki_th = 0.5

        # print control gains to terminal        
        print('DC_gain', DC_gain)
        print('kp_th: ', self.kp_th)
        print('kd_th: ', self.kd_th)
        print('kp_z: ', self.kp_z)
        print('kd_z: ', self.kd_z)

        self.integrator_z = 0.
        self.error_z_d1 = 0.
        self.z_dot = P.zdot0
        self.z_d1 = P.z0
        self.theta_dot = P.thetadot0
        self.theta_d1 = P.theta0
        self.integrator_th = 0.
        self.error_th_d1 = 0.   


    def update(self, z_r, state):
        z = state[0][0]
        theta = state[1][0]
        
        error_z = z_r - z

        self.zdot = (2.0*self.beta - P.Ts) / (2.0*self.beta + P.Ts) * self.z_dot \
            + (2.0 / (2.0*self.beta + P.Ts)) * ((z - self.z_d1))
        
        if np.abs(self.z_dot) < 0.07:
            self.integrator_z = self.integrator_z \
                + (P.Ts / 2) * (error_z + self.error_z_d1)

        # the reference angle for theta comes from the
        # outer loop PD control
        theta_r = self.kp_z * (z_r - z) - self.kd_z * self.zdot # self.ki_z * self.integrator_z

        # self.theta_max = 90.0 * np.pi / 180.0

        # theta_r = saturate(theta_r, self.theta_max)

        error_th = theta_r - theta
        # differentiate theta
        self.thetadot = (2.0*self.sigma - P.Ts) / (2.0*self.sigma + P.Ts) * self.theta_dot \
            + (2.0 / (2.0*self.sigma + P.Ts)) * ((theta - self.theta_d1))
        
       
        
        F_tilde =  self.kp_th * (error_th) - self.kd_th * self.thetadot
        F_fl = P.m1 * P.g * (z / P.length) + P.m2 * P.g / 2.0 

        F_unsat = F_tilde + F_fl

        F = saturate(F_unsat, P.F_max)

        self.error_z_d1 = error_z
        self.z_d1 = z
        self.error_th_d1 = error_th
        self.theta_d1 = theta

        return F
    

class ctrlPID2:
    def __init__(self):
        # dirty derivative parameters
        self.sigma = 0.05  # cutoff freq for dirty derivative
        self.beta = (2 * self.sigma - P.Ts) / (2 * self.sigma + P.Ts)  # dirty derivative gain

        ####################################################
        #       PD Control: Time Design Strategy
        ####################################################
        # tuning parameters
        tr_z = 1.2  # rise time for outer loop - first part of problem
        zeta_z = 0.707  # damping ratio for outer loop
        M = 10  # time scale separation between inner and outer loop
        zeta_th  = 0.707  # damping ratio for inner loop
        self.ki_z = -0.1  # integral gain on outer loop

        # saturation limits
        #self.theta_max = 10.0 * np.pi / 180.0  # Max theta, rads
        #self.theta_max = 45.0 * np.pi / 180.0  # Max theta, rads

        #---------------------------------------------------
        #                    Inner Loop
        #---------------------------------------------------
        ze = P.length/2.0  # equilibrium position - center of beam
        b0 = P.length/(P.m2*P.length**2/3.0+P.m1*ze**2)
        tr_theta = tr_z/M  # rise time for inner loop
        wn_th = 2.2/tr_theta  # natural frequency for inner loop
        self.kp_th = wn_th**2/b0  # kp - inner
        self.kd_th = 2.0*zeta_th*wn_th/b0  # kd - inner

        # DC gain for inner loop
        DC_gain = 1.0

        #---------------------------------------------------
        #                    Outer Loop
        #---------------------------------------------------
        wn_z = 2.2/tr_z  # natural frequency - outer loop
        self.kp_z = -wn_z**2/P.g/DC_gain  # kp - outer
        self.kd_z = -2.0*zeta_z*wn_z/P.g/DC_gain  # kd - outer

        # print control gains to terminal        
        print('DC_gain', DC_gain)
        print('kp_th: ', self.kp_th)
        print('kd_th: ', self.kd_th)
        print('kp_z: ', self.kp_z)
        print('ki_z: ', self.ki_z)
        print('kd_z: ', self.kd_z)

        #---------------------------------------------------
        # initialize variables needed for integrator and differentiators
        #---------------------------------------------------
        self.integrator_z = 0.
        self.error_z_d1 = 0.
        self.z_dot = 0.
        self.z_d1 = 0.
        self.theta_dot = 0.
        self.theta_d1 = 0.

    def update(self, z_r, y):
        z = y[0][0]
        theta = y[1][0]

        #---------------------------------------------------
        # Update Outer Loop (z-control)
        #---------------------------------------------------
        # Compute the error in z
        error_z = z_r - z
        
        # differentiate z
        self.z_dot = self.beta * self.z_dot + (1 - self.beta) * ((z - self.z_d1) / P.Ts)

        # integrate error in z_dot is less than a certain value: 
        if np.abs(self.z_dot) <= 0.1:
            self.integrator_z = self.integrator_z + (P.Ts / 2) * (error_z + self.error_z_d1)

        # PID control 
        theta_r = self.kp_z * error_z + self.ki_z * self.integrator_z - self.kd_z * self.z_dot


        #---------------------------------------------------
        # Update Inner Loop (theta-control)
        #---------------------------------------------------
        # Compute the error in theta
        error_th = theta_r - theta

        # differentiate theta
        self.theta_dot = self.beta * self.theta_dot + (1 - self.beta) * ((theta - self.theta_d1) / P.Ts)

        # PD control on theta
        F_tilde = self.kp_th * error_th - self.kd_th * self.theta_dot

        # feedback linearizing force
        F_fl = P.m1*P.g*(z/P.length) + P.m2*P.g/2.0

        # saturate the force
        F = saturate(F_tilde + F_fl, P.F_max)

        # update variables
        self.error_z_d1 = error_z
        self.z_d1 = z
        self.theta_d1 = theta

        # return computed force
        return F


def saturate(u, limit):
    if abs(u) > limit:
        u = limit * np.sign(u)
    return u






