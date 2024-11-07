import numpy as np
import VTOLParam as P

class ctrlPD:
    def __init__(self):
        ####################################################
        #       PD Control: Time Design Strategy
        ####################################################
        # tuning parameters
        tr_h = 4.0
        zeta_h = 0.707
        tr_z = 4.0
        zeta_z = 0.707
        tr_th = 8          # Rise time for inner loop (theta)
        zeta_th = 0.707       # inner loop Damping Coefficient

        #---------------------------------------------------
        #                    Inner Loop
        #---------------------------------------------------
        # parameters of the open loop transfer function
        beta = 1/(P.Jc + 2*P.mr*P.d**2)

        # coefficients for desired inner loop
        wn_th = 2.2 / tr_th     # Natural frequency

        # compute gains
        self.kp_th = wn_th**2/beta
        self.kd_th = (2.0 * zeta_th * wn_th) / beta
        DC_gain = self.kp_th / (beta * self.kp_th)
        #---------------------------------------------------
        #                    Outer Loop
        #---------------------------------------------------
        # coefficients for desired outer loop
        M = 10         # Time scale separation 
        zeta_z = 0.707     # outer loop Damping Coefficient
        tr_z = M * tr_th   # desired rise time, s
        wn_z = 2.2 / tr_z  # desired natural frequency

        # compute gains
        
        beta = P.mu/(P.mc+2*P.mr)

        self.kd_z = 2*wn_z*zeta_z - beta / 1
        self.kp_z = wn_z**2

        # print control gains to terminal        
        print('DC_gain', DC_gain)
        print('kp_th: ', self.kp_th)
        print('kd_th: ', self.kd_th)
        print('kp_z: ', self.kp_z)
        print('kd_z: ', self.kd_z)

        #---------------------------------------------------
        #                    zero canceling filter
        #---------------------------------------------------
        self.filter = zeroCancelingFilter(DC_gain)

    def update(self, z_r, state):
        z = state[0][0]
        theta = state[1][0]
        zdot = state[2][0]
        thetadot = state[3][0]

        # the reference angle for theta comes from the
        # outer loop PD control
        tmp = self.kp_z * (z_r - z) - self.kd_z * zdot

        # # low pass filter the outer loop to cancel
        # # left-half plane zero and DC-gain
        # theta_r = self.filter.update(tmp)

        # the force applied to the cart comes from the
        # inner loop PD control
        F = self.kp_th * (tmp - theta) - self.kd_th * thetadot

        return F

class zeroCancelingFilter:
    def __init__(self, DC_gain):
        self.a = -3.0 / (2.0 * P.length * DC_gain)
        self.b = np.sqrt(3.0 * P.g / (2.0 * P.length))
        self.state = 0.0

    def update(self, input):
        # integrate using RK1
        self.state = self.state + P.Ts * (-self.b * self.state + self.a * input)
        return self.state






