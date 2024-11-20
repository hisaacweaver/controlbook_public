import numpy as np
import blockbeamParam as P
import control as cnt


class ctrlObserver:
    def __init__(self):
        self.integrator_z = 0.0  # integrator
        self.error_z_d1 = 0.0  # error signal delayed by 1 sample
        self.Ts = P.Ts  # sample rate of controller

        # tuning parameters
        tr_z = 1.2        # rise time for position
        tr_theta = 0.5    # rise time for angle
        zeta_z = 0.95  # damping ratio position
        zeta_th = 0.95  # damping ratio angle
        integrator_pole = -5.0

        # State Space Equations
        # xdot = A*x + B*u
        # y = C*x
        A = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, -P.g, 0.0, 0.0],
            [-P.m1*P.g/((P.m2*P.length**2)/3.0+P.m1*(P.length/2.0)**2), \
             0.0, 0.0, 0.0]])
        B = np.array([[0.0],
                      [0.0],
                      [0.0],
                      [P.length / (P.m2 * P.length ** 2 / 3.0 \
                            + P.m1 * P.length ** 2 / 4.0)]])
        C = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0]])
        self.Cr = np.array([[1.0, 0.0, 0.0, 0.0]])        

        # form augmented system
        A1 = np.vstack((
                np.hstack((A, np.zeros((4,1)))),
                np.hstack((-self.Cr, np.zeros((1,1))))))
        B1 = np.vstack((B, np.zeros((1,1))))

        # gain calculation
        wn_th = 2.2 / tr_theta
        wn_z = 2.2 / tr_z
        des_char_poly = np.convolve(
            np.convolve([1, 2*zeta_z*wn_z, wn_z**2],
                        [1, 2*zeta_th*wn_th, wn_th**2]),
            [1, -integrator_pole])
        des_poles = np.roots(des_char_poly)

        # Compute the gains if the system is controllable
        if np.linalg.matrix_rank(cnt.ctrb(A1, B1)) != 5:
            print("The system is not controllable")
        else:
            K1 = cnt.place(A1, B1, des_poles)
            self.K = K1[0][0:4]
            self.ki = K1[0][4]
        print('K: ', self.K)
        print('ki: ', self.ki)

    def update(self, z_r, x):
        z = self.Cr @ x

        # calc error
        error_z = z_r - z

        # integrate error
        self.integrator_z = self.integrator_z \
            + (P.Ts / 2.0) * (error_z + self.error_z_d1)
        self.error_z_d1 = error_z

        # Construct the linearized state
        x_tilde = x - np.array([[P.ze], [0], [0], [0]])

        # equilibrium force
        F_e = P.m1*P.g*P.ze/P.length + P.m2*P.g/2.0

        # Compute the state feedback controller
        F_tilde = -self.K @ x_tilde - self.ki*self.integrator_z
        F_unsat = F_e + F_tilde
        F = saturate(F_unsat[0][0], P.F_max)
        self.integratorAntiWindup(F, F_unsat)
        
        return F

    def integratorAntiWindup(self, F, F_unsat):
        if self.ki != 0.0:
            self.integrator_z = self.integrator_z + P.Ts/self.ki*(F-F_unsat)


def saturate(u, limit):
    if abs(u) > limit:
        u = limit * np.sign(u)
    return u

