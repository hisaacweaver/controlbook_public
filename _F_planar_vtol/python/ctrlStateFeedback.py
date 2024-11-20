import numpy as np
import control as cnt
import VTOLParam as P

class ctrlStateFeedback:
    def __init__(self):
        #--------------------------------------------------
        # State Feedback Control Design
        #--------------------------------------------------
        # tuning parameters
        tr_h = 4.0  
        zeta_h = 0.707

        tr_z = 8.0
        zeta_z = 0.707  # damping ratio position 

        tr_theta = tr_z/10
        zeta_theta = 0.707  # damping ratio angle   
        
        wn_h = 2.2/tr_h
        wn_z = 2.2/tr_z
        wn_theta = 2.2/tr_theta

        des_char_poly_lat = np.convolve([1, 2*zeta_z*wn_z, wn_z**2], [1, 2*zeta_theta*wn_theta, wn_theta**2])

        des_char_poly_lon = [1, 2*zeta_h*wn_h, wn_h**2]

        des_poles_lat = np.roots(des_char_poly_lat)
        des_poles_lon = np.roots(des_char_poly_lon)

        A_lat = np.array([[0.0, 0.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0],
                          [0.0, -P.Fe/(P.mc+2*P.mr), -P.mu/(P.mc+2*P.mr), 0.0],
                          [0.0, 0.0, 0.0, 0.0]])
        
        B_lat = np.array([[0.0],
                          [0.0],
                          [0.0],
                          [1/(P.Jc+2*P.mr*P.d**2)]])
        
        C_lat = np.array([[1.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0]])
                          

        A_lon = np.array([[0.0, 1.0],
                          [0.0, 0.0]])
        
        B_lon = np.array([[0.0],
                          [1/(P.mc+2*P.mr)]])
        
        C_lon = np.array([[1.0, 0.0]])

        if np.linalg.matrix_rank(cnt.ctrb(A_lat, B_lat)) != 4:
            print("The lateral system is not controllable")
        else:
            Cr = np.array([[1.0, 0.0, 0.0, 0.0]])
            self.K_lat = cnt.place(A_lat, B_lat, des_poles_lat)
            self.kr_lat = -1.0/(Cr @ np.linalg.inv(A_lat-B_lat @ self.K_lat) @ B_lat)
        # print gains to terminal
        print('K_lat: ', self.K_lat)
        print('kr_lat: ', self.kr_lat)

        if np.linalg.matrix_rank(cnt.ctrb(A_lon, B_lon)) != 2:
            print("The longitudinal system is not controllable")
        else:
            Cr = np.array([[1.0, 0.0]])
            self.K_lon = cnt.place(A_lon, B_lon, des_poles_lon)
            self.kr_lon = -1.0/(Cr @ np.linalg.inv(A_lon-B_lon @ self.K_lon) @ B_lon)
        # print gains to terminal
        print('K_lon: ', self.K_lon)
        print('kr_lon: ', self.kr_lon)

        self.ki_h = -0.1
        self.ki_z = -0.1
        self.integrator_h = 0.0
        self.integrator_z = 0.0
        self.error_d1_h = 0.0
        self.error_d1_z = 0.0
    

    def update(self, z_r, h_r, x):
        x_lon = np.array([[x[1][0]], [x[4][0]]])
        x_lat = np.array([[x[0][0]], [x[2][0]], [x[3][0]], [x[5][0]]])

        z_error = z_r - x[0][0]
        h_error = h_r - x[2][0]

        self.integrator_h = self.integrator_h \
                          + (P.Ts / 2.0) * (h_error + self.error_d1_h)
        self.error_d1_h = h_error

        self.integrator_z = self.integrator_z \
                          + (P.Ts / 2.0) * (z_error + self.error_d1_z)
        self.error_d1_z = z_error

        F_eq = P.Fe
        tau_eq = 0.0
        T_tilde = -self.K_lat @ x_lat + self.kr_lat * h_r #+ self.ki_h * self.integrator_h
        F_tilde = -self.K_lon @ x_lon + self.kr_lon * z_r #+ self.ki_z * self.integrator_z

        F = F_eq + F_tilde[0][0]
        tau = tau_eq + T_tilde[0][0]
        u = np.array([[F], [tau]])
        tau_unsat = P.mixing @ u
        tau = self.saturate(tau_unsat)

        return tau
    
    def saturate(self, u):
        for i in range(len(u)):
            if abs(u[i][0]) > P.max_thrust:
                u[i][0] = P.max_thrust * np.sign(u[i][0])
        return u


import hw11_planarVTOLSIM