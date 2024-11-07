import numpy as np
import hummingbirdParam as P

class ctrlPD:
    def __init__(self):

        # Design parameters for PD control
        tr_theta = 0.5
        zeta_theta = 0.707

        # Calculate natural frequency
        wn = 2.2 / tr_theta
        b_theta = P.ellT/(P.m1 * P.ell1**2 + P.m2 * P.ell2**2 + P.J1y + P.J2y)

        # PD gains based on the characteristic polynomial
        self.kp_theta = 1.0 #wn**2 / b_theta  # b_theta is a system parameter
        self.kd_theta = 0.75 #2 * zeta_theta * wn / b_theta
        self.ki_theta = 2
        # Design parameters for PD control
        tr_psi = 1.0
        zeta_psi = 0.707

        M = 10
        tr_phi = tr_psi / M
        zeta_phi = 0.707
        
        wn_phi = 2.2 / tr_phi

        self.kp_phi = wn_phi**2 * P.J1x
        self.kd_phi = 2 * zeta_phi * wn_phi * P.J1x

        wn_psi = wn_phi / M

        force_equilibrium = (P.m1*P.ell1 + P.m2*P.ell2)*P.g / P.ellT

        JT = P.m1*P.ell1**2 + P.m2*P.ell2**2 + P.J2z + P.m3*(P.ell3x**2 + P.ell3y**2)

        beta_psi = P.ellT * force_equilibrium / (JT + P.J1z)

        self.kp_psi = wn_psi**2 / beta_psi
        self.kd_psi = 2 * zeta_psi * wn_psi / beta_psi

        print(f"\n\nTheta Gains \nkp:{self.kp_theta} \nkd:{self.kd_theta} \nki: {self.ki_theta}\n\n")
        print(f"Phi Gains \nkp:{self.kp_phi} \nkd:{self.kd_phi}\n")
        print(f"Psi Gains \nkp:{self.kp_psi} \nkd:{self.kd_psi}\n\n")

        self.theta_dot = P.thetadot0  # estimated derivative of theta
        self.theta_d1 = P.theta0  # theta delayed by one sample
        self.theta_error_dot = 0.0  # estimated derivative of error
        self.theta_error_d1 = 0.0  # Error delayed by one sample
        self.theta_integrator = 0.0  # integrator




    def update(self, x, theta_ref, psi_ref):
        theta = x[1][0]
        thetadot = x[4][0]
        phi = x[0][0]
        phidot = x[3][0]
        psi = x[2][0]
        psidot = x[5][0]

        
        


        theta_error = theta_ref - theta
        if abs(thetadot < 0.1):
            self.theta_integrator = self.theta_integrator \
                + (P.Ts / 2) * (theta_error + self.theta_error_d1)

        force_equilibrium = (P.m1*P.ell1 + P.m2*P.ell2)*P.g / P.ellT * np.cos(theta)
        force_tilde = self.kp_theta * theta_error - self.kd_theta * thetadot + self.ki_theta * self.theta_integrator
        force = force_equilibrium + force_tilde
        
        psi_error = psi_ref - psi
        phi_c = self.kp_psi * psi_error - self.kd_psi * psidot

        phi_error = phi_c - phi
        torque = self.kp_phi * phi_error - self.kd_phi * phidot
        
        # convert force and torque to pwm signals
        pwm_left = 1/(2*P.km) * (force + torque/P.d)
        pwm_right = 1/(2*P.km) * (force - torque/P.d)
        pwm = np.array([[pwm_left], [pwm_right]])
        pwm = saturate(pwm, 0, 1) 
        return pwm, np.array([[phi_c, theta_ref, psi_ref]])


def saturate(u, low_limit, up_limit):
    if isinstance(u, float) is True:
        u = np.max((np.min((u, up_limit)), low_limit))
    else:
        for i in range(0, u.shape[0]):
            u[i][0] = np.max((np.min((u[i][0], up_limit)), low_limit))
    return u

import lab6_hummingbirdSim




