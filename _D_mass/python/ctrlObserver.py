import numpy as np
import control as cnt
import massParam as P

class ctrlObserver:
    # dirty derivatives to estimate thetadot
    def __init__(self):
        #  tuning parameters
        tr = 0.4
        zeta = 0.707

        tr_obs = tr/10
        zeta_obs = 0.707

        # State Space Equations
        # xdot = A*x + B*u
        # y = C*x

        self.A = np.array([[0.0, 1.0],
                      [-P.k/P.m, -P.b/P.m]])
        self.B = np.array([[0.0],
                      [1/P.m]])        
        self.C = np.array([[1.0, 0.0]])

        # gain calculation
        wn = 2.2 / tr  # natural frequency
        des_char_poly = [1, 2 * zeta * wn, wn**2]
        des_poles = np.roots(des_char_poly)
        des_poles = [-15, -15.1]

        # Compute the gains if the system is controllable
        if np.linalg.matrix_rank(cnt.ctrb(self.A, self.B)) != 2:
            print("The system is not controllable")
        else:
            self.K = (cnt.place(self.A, self.B, des_poles))
            self.kr = -1.0 / (self.C @ np.linalg.inv(self.A - self.B @ self.K) @ self.B)

        self.ki = -2.0
        self.integrator = 0.0
        self.error_d1 = 0.0

        # observer design
        wn_obs = 2.2 / tr_obs
        des_obsv_char_poly = [1, 2*zeta_obs*wn_obs, wn_obs**2]
        des_obsv_poles = np.roots(des_obsv_char_poly)
        # Compute the gains if the system is controllable
        if np.linalg.matrix_rank(cnt.ctrb(self.A.T, self.C.T)) != 2:
            print("The system is not observerable")
        else:
            self.L = cnt.acker(self.A.T, self.C.T, des_obsv_poles).T
        print('L^T: ', self.L.T)

        self.x_hat = np.array([
            [0.0],  # theta_hat_0
            [0.0],  # thetadot_hat_0
        ])
        self.tau_d1 = 0.0  # control torque, delayed 1 sample
        
        print('K: ', self.K)
        print('kr: ', self.kr)
        print(des_poles)

    def update(self, z_r, x):

        xhat = self.update_observer(x)
        zhat = xhat[0][0]

        error = z_r - zhat

        self.integrator = self.integrator \
                          + (P.Ts / 2.0) * (error + self.error_d1)
        self.error_d1 = error

        # Compute the state feedback controller
        f_tilde = -self.K @ xhat + - self.ki * self.integrator + self.kr * z_r

        # compute total torque
        tau = saturate(f_tilde[0][0], P.F_max)
        self.tau_d1 = tau
        return tau, xhat

    def update_observer(self, x_m):
        # update the observer using RK4 integration
        F1 = self.observer_f(self.x_hat, x_m)
        F2 = self.observer_f(self.x_hat + P.Ts / 2 * F1, x_m)
        F3 = self.observer_f(self.x_hat + P.Ts / 2 * F2, x_m)
        F4 = self.observer_f(self.x_hat + P.Ts * F3, x_m)
        self.x_hat = self.x_hat + P.Ts / 6 * (F1 + 2*F2 + 2*F3 + F4)
        return self.x_hat

    def observer_f(self, x_hat, x_m):
        xhat_dot = self.A @ x_hat\
                   + self.B * (self.tau_d1)\
                   + self.L * (x_m - self.C @ x_hat)
        return xhat_dot


def saturate(u, limit):
    if abs(u) > limit:
        u = limit * np.sign(u)
    return u


if __name__ == "__main__":
    import hw13_massSim
