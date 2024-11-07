import numpy as np
import control as cnt
import blockbeamParam as P

class ctrlStateFeedback:
    def __init__(self):
        #--------------------------------------------------
        # State Feedback Control Design
        #--------------------------------------------------
        # tuning parameters
        tr_theta = 0.05       
        tr_z = 0.2
        zeta_z = 0.707  # damping ratio position
        zeta_th = 0.707  # damping ratio angle
        
        # State Space Equations
        # xdot = A*x + B*u
        # y = C*x

        z_e = 0.0

        A = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, -P.g, 0.0, 0.0],
            [-P.m1*P.g/((P.m2*P.length**2 / 3.0) + P.m1*z_e**2), 0.0, 0.0, 0.0]])
        
        B = np.array([[0.0],
                      [0.0],
                      [0.0],
                      [P.length/((P.m2*P.length**2 / 3.0) + P.m1*z_e**2)]])
        C = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0]])
        
        # gain calculation
        wn_th = 2.2 / tr_theta  # natural frequency for angle
        wn_z = 2.2 / tr_z  # natural frequency for position
        
        des_char_poly = np.convolve(
            [1, 2 * zeta_z * wn_z, wn_z**2],
            [1, 2 * zeta_th * wn_th, wn_th**2])
        des_poles = np.roots(des_char_poly)

        print(des_poles)

        if np.linalg.matrix_rank(cnt.ctrb(A, B)) != 4:
            print("The system is not controllable")
        else:
            self.K = cnt.place(A, B, des_poles)
            # self.K = np.array([50, 6.05, -0.05, -0.105])
            Cr = np.array([[1.0, 0.0, 0.0, 0.0]])
            self.kr = -1.0 / (Cr @ np.linalg.inv(A-B @ self.K) @ B)
        # print gains to terminal
        print('K: ', self.K)
        print('kr: ', self.kr)

    def update(self, z_r, x):
        z = x[0][0]
        F_unsat = -self.K @ x + self.kr * z_r
        F_fl = P.m1 * P.g * (0 / P.length) + P.m2 * P.g / 2.0
        F = F_unsat + F_fl
        F = saturate(F_unsat[0][0], 10)
        return F


def saturate(u, limit):
    if abs(u) > limit:
        u = limit * np.sign(u)
    return u


if __name__ == "__main__":
    import hw11_blockbeamSim