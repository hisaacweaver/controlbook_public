import numpy as np
import control as cnt
import VTOLParam as P

class ctrlObserver:
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

        self.A_lat = np.array([[0.0, 0.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0],
                          [0.0, -P.Fe/(P.mc+2*P.mr), -P.mu/(P.mc+2*P.mr), 0.0],
                          [0.0, 0.0, 0.0, 0.0]])
        
        self.B_lat = np.array([[0.0],
                          [0.0],
                          [0.0],
                          [1/(P.Jc+2*P.mr*P.d**2)]])
        
        self.C_lat = np.array([[1.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0]])
                          

        self.A_lon = np.array([[0.0, 1.0],
                          [0.0, 0.0]])
        
        self.B_lon = np.array([[0.0],
                          [1/(P.mc+2*P.mr)]])
        
        self.C_lon = np.array([[1.0, 0.0]])

        if np.linalg.matrix_rank(cnt.ctrb(self.A_lat, self.B_lat)) != 4:
            print("The lateral system is not controllable")
        else:
            Cr = np.array([[1.0, 0.0, 0.0, 0.0]])
            self.K_lat = cnt.place(self.A_lat, self.B_lat, des_poles_lat)
            self.kr_lat = -1.0/(Cr @ np.linalg.inv(self.A_lat-self.B_lat @ self.K_lat) @ self.B_lat)
        # print gains to terminal
        print('K_lat: ', self.K_lat)
        print('kr_lat: ', self.kr_lat)

        if np.linalg.matrix_rank(cnt.ctrb(self.A_lon, self.B_lon)) != 2:
            print("The longitudinal system is not controllable")
        else:
            Cr = np.array([[1.0, 0.0]])
            self.K_lon = cnt.place(self.A_lon, self.B_lon, des_poles_lon)
            self.kr_lon = -1.0/(Cr @ np.linalg.inv(self.A_lon-self.B_lon @ self.K_lon) @ self.B_lon)
        # print gains to terminal
        print('K_lon: ', self.K_lon)
        print('kr_lon: ', self.kr_lon)


        #--------------------------------------------------
        # Observer Design
        tr_obs = tr_theta/10
        zeta_obs = 0.707
        wn_obs = 2.2 / tr_obs
        des_obsv_char_poly = [1, 2*zeta_obs*wn_obs, wn_obs**2]
        des_obsv_poles = np.roots(des_obsv_char_poly)
        if np.linalg.matrix_rank(cnt.ctrb(self.A_lon.T, self.C_lon.T)) != 2:
            print("The system is not observerable")
        else:
            self.L_lon = cnt.acker(self.A_lon.T, self.C_lon.T, des_obsv_poles).T
        print('L^T: ', self.L_lon.T)

        tr_obs = tr_theta/10
        zeta_obs = 0.707
        wn_obs = 2.2 / tr_obs
        des_obsv_char_poly = [1, 2*zeta_obs*wn_obs, wn_obs**2]
        des_obsv_poles = np.roots(des_obsv_char_poly)
        wn_z_obs = 10.0 * wn_z
        wn_th_obs = 10.0 * wn_theta

        # compute observer gains
        des_obs_char_poly = np.convolve(
            [1, 2 * zeta_z * wn_z_obs, wn_z_obs**2],
            [1, 2 * zeta_theta * wn_th_obs, wn_th_obs**2])
        des_obs_poles = np.roots(des_obs_char_poly)
        if np.linalg.matrix_rank(cnt.ctrb(self.A_lat.T, self.C_lat.T)) != 4:
            print("The system is not observerable")
        else:
            self.L_lat = cnt.place(self.A_lat.T, self.C_lat.T, des_obs_poles).T
        print('L^T: ', self.L_lat.T)

        self.x_hat = np.array([
            [0.0],  # z_hat_0
            [0.0],  # h_hat_0
            [0.0],  # theta_hat_0
            [0.0],  # zdot_hat_0
            [0.0],  # hdot_hat_0
            [0.0],  # thetadot_hat_0
        ])

        self.xhat_lon = np.array([[self.x_hat[1][0]], [self.x_hat[4][0]]])
        self.xhat_lat = np.array([[self.x_hat[0][0]], [self.x_hat[2][0]], [self.x_hat[3][0]], [self.x_hat[5][0]]])

        self.tau_d1 = 0.0  # control torque, delayed 1 sample
        
        self.ki_h = -0.1
        self.ki_z = -0.1
        self.integrator_h = 0.0
        self.integrator_z = 0.0
        self.error_d1_h = 0.0
        self.error_d1_z = 0.0
    

    def update(self, z_r, h_r, x):
        x_lon = np.array([[x[1][0]], [x[4][0]]])
        x_lat = np.array([[x[0][0]], [x[2][0]], [x[3][0]], [x[5][0]]])

        # xhat_lon = self.update_observer_lon(x_lon)
        # xhat_lat = self.update_observer_lat(x_lat)
        

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
        F = F_eq + F_tilde
        tau = tau_eq + T_tilde
        u = np.array([[F[0][0]], [tau[0][0]]])
        tau_unsat = P.mixing @ u
        tau = self.saturate(tau_unsat)
        self.tau_d1 = tau
        return tau, x_lat, x_lon
    
    def update_observer_lon(self, y_m):
        # update the observer using RK4 integration
        F1 = self.observer_lon(self.xhat_lon, y_m)
        F2 = self.observer_lon(self.xhat_lon + P.Ts / 2 * F1, y_m)
        F3 = self.observer_lon(self.xhat_lon + P.Ts / 2 * F2, y_m)
        F4 = self.observer_lon(self.xhat_lon + P.Ts * F3, y_m)
        self.xhat_lon = self.xhat_lon + P.Ts / 6 * (F1 + 2*F2 + 2*F3 + F4)
        return self.xhat_lon

    def observer_lon(self, x_hat, y_m):
        print(x_hat)
        print(y_m)
        print(self.C_lon @ x_hat)
        xhat_dot = self.A_lon @ x_hat\
                   + self.B_lon * (self.tau_d1)\
                   + self.L_lon * (y_m - self.C_lon @ x_hat)
        return xhat_dot

    def update_observer_lat(self, y_m):
        # update the observer using RK4 integration
        F1 = self.observer_lat(self.xhat_lat, y_m)
        F2 = self.observer_lat(self.xhat_lat + P.Ts / 2 * F1, y_m)
        F3 = self.observer_lat(self.xhat_lat + P.Ts / 2 * F2, y_m)
        F4 = self.observer_lat(self.xhat_lat + P.Ts * F3, y_m)
        self.xhat_lat = self.xhat_lat + P.Ts / 6 * (F1 + 2*F2 + 2*F3 + F4)
        return self.xhat_lat

    def observer_lat(self, x_hat, y_m):
        print(x_hat)
        print(y_m)
        print(self.C_lat @ x_hat)
        xhat_dot = self.A_lat @ x_hat\
                   + self.B_lat * (self.tau_d1)\
                   + self.L_lat * (y_m - self.C_lat @ x_hat)
        return xhat_dot
    
    def saturate(self, u):
        for i in range(len(u)):
            if abs(u[i][0]) > P.max_thrust:
                u[i][0] = P.max_thrust * np.sign(u[i][0])
        return u


import hw13_planarVTOLSIM