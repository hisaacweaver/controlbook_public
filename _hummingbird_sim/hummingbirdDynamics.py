import numpy as np 
import hummingbirdParam as P
from numpy import sin, cos


class HummingbirdDynamics:
    def __init__(self, initial_state=np.zeros((6,1)), alpha=0.0):

        # Initial state conditions
        self.state = initial_state
        # vary the actual physical parameters
        self.ell1 = P.ell1 * (1.+alpha*(2.*np.random.rand()-1.))
        self.ell2 = P.ell2 * (1.+alpha*(2.*np.random.rand()-1.))
        self.ell3x = P.ell3x * (1.+alpha*(2.*np.random.rand()-1.))
        self.ell3y = P.ell3y * (1.+alpha*(2.*np.random.rand()-1.))
        self.ell3z = P.ell3z * (1.+alpha*(2.*np.random.rand()-1.))
        self.ellT = P.ellT * (1.+alpha*(2.*np.random.rand()-1.))
        self.d = P.d * (1.+alpha*(2.*np.random.rand()-1.))
        self.m1 = P.m1 * (1.+alpha*(2.*np.random.rand()-1.))
        self.m2 = P.m2 * (1.+alpha*(2.*np.random.rand()-1.))
        self.m3 = P.m3 * (1.+alpha*(2.*np.random.rand()-1.))
        self.J1x = P.J1x * (1.+alpha*(2.*np.random.rand()-1.))
        self.J1y = P.J1y * (1. + alpha * (2. * np.random.rand() - 1.))
        self.J1z = P.J1z * (1. + alpha * (2. * np.random.rand() - 1.))
        self.J2x = P.J2x * (1.+alpha*(2.*np.random.rand()-1.))
        self.J2y = P.J2y * (1. + alpha * (2. * np.random.rand() - 1.))
        self.J2z = P.J2z * (1. + alpha * (2. * np.random.rand() - 1.))
        self.J3x = P.J3x * (1.+alpha*(2.*np.random.rand()-1.))
        self.J3y = P.J3y * (1. + alpha * (2. * np.random.rand() - 1.))
        self.J3z = P.J3z * (1. + alpha * (2. * np.random.rand() - 1.))
        self.km = P.km * (1. + alpha * (2. * np.random.rand() - 1.))
 
    def update(self, u: np.ndarray):
        # This is the external method that takes the input u at time
        # t and returns the output y at time t.
        # saturate the input force
        u = saturate(u, P.torque_max)
        self.rk4_step(u)  # propagate the state by one time sample
        y = self.h()  # return the corresponding output
        return y

    def f(self, state: np.ndarray, pwms: np.ndarray):
        # Return xdot = f(x,u)
        phidot = state[3][0]
        thetadot = state[4][0]
        psidot = state[5][0]
        pwm_left = pwms[0][0]
        pwm_right = pwms[1][0]

        # The equations of motion go here
        M = self._M(state)
        C = self._C(state)
        partialP = self._partialP(state)

        force = self.km * (pwm_left + pwm_right)
        torque = self.d * self.km * (pwm_left - pwm_right)
        tau = self._tau(state, force, torque)
        B = self._B()

        qddot = np.linalg.inv(M) @ (-C - partialP + tau - B @ state[3:6])
        
        phiddot = qddot[0][0]
        thetaddot = qddot[1][0]
        psiddot = qddot[2][0]
        
        # build xdot and return
        xdot = np.array([[phidot],
                         [thetadot],
                         [psidot],
                         [phiddot],
                         [thetaddot],
                         [psiddot]])
        return xdot

    def h(self):
        # FIXME Fill in this function
        # return y = h(x)
        phi = self.state[0][0]
        theta = self.state[1][0]
        psi = self.state[2][0]
        y = np.array([[phi], [theta], [psi]])
        return y

    def rk4_step(self, u: np.ndarray):
        # Integrate ODE using Runge-Kutta RK4 algorithm
        F1 = self.f(self.state, u)
        F2 = self.f(self.state + P.Ts / 2 * F1, u)
        F3 = self.f(self.state + P.Ts / 2 * F2, u)
        F4 = self.f(self.state + P.Ts * F3, u)
        self.state = self.state + P.Ts / 6 * (F1 + 2*F2 + 2*F3 + F4)

    def _M(self, state: np.ndarray):
        # FIXME Fill in this function
        phi = state[0][0]
        theta = state[1][0]
        psi = state[2][0]
        phidot = state[3][0]
        thetadot = state[4][0]
        psidot = state[5][0]

        J1x = self.J1x
        J1y = self.J1y
        J1z = self.J1z

        J2x = self.J2x
        J2y = self.J2y
        J2z = self.J2z

        J3z = self.J3z

        m1 = self.m1
        m2 = self.m2
        m3 = self.m3
        l1 = self.ell1
        l2 = self.ell2
        l3x = self.ell3x
        l3y = self.ell3y

        # Fill out M22, M23, and M33
        M22 = m1*l1**2 + m2*l2**2 + J2y + J1y * cos(phi)**2 + J1z * sin(phi)**2
        M23 = (J1y - J1z) * sin(phi) * cos(phi) * cos(theta)
        M33 = (m1*l1**2 + m2*l2**2 + J2z + J1y * sin(phi)**2 + J1z * cos(phi)**2) * cos(theta)**2 + (J1x + J2x)*sin(theta)**2 + m3 * (l3x**2 + l3y**2) + J3z
        M33 = (m1*l1**2 + m2*l2**2 + J2z + J1y * sin(phi)**2 + J1z * cos(phi)**2) * cos(theta)**2 + (J1x + J2x)*sin(theta)**2 + m3 * (l3x**2 + l3y**2) + J3z
        corners = -J1x*sin(theta)
        # Return the M matrix
        return np.array([[J1x, 0, corners],
                      [0, M22, M23],
                      [corners, M23, M33]
                      ])

    def _C(self, state: np.ndarray):
        # FIXME Fill in this function
        #extact any necessary variables from the state
        phi = state[0][0]
        theta = state[1][0]
        psi = state[2][0]
        phidot = state[3][0]
        thetadot = state[4][0]
        psidot = state[5][0]

        s_phi = sin(phi)
        c_phi = cos(phi)
        s_theta = sin(theta)
        c_theta = cos(theta)

        J1x = self.J1x
        J1y = self.J1y
        J1z = self.J1z

        J2x = self.J2x
        J2y = self.J2y
        J2z = self.J2z

        m1 = self.m1
        m2 = self.m2
        l1 = self.ell1
        l2 = self.ell2
        

        N33 = 2*(J1x + J2x - m1*l1**2 - m2*l2**2 - J2z - J1y*s_phi**2 - J1z*c_phi**2)*s_theta*c_theta

        c3_1 = thetadot**2 * (J1z-J1y) * s_phi*c_phi*s_theta
        c3_2 = ((J1y-J1z)*(c_phi**2-s_phi**2) - J1x) * c_theta*phidot*thetadot
        c3_3 = (J1z-J1y)*s_phi*c_phi*s_theta*thetadot**2 + 2*(J1y-J1z)*s_phi*c_phi*phidot*psidot
        c3_4 = 2*(-m1*l1**2 - m2*l2**2 - J2z + J1x + J2x + J1y*s_phi**2 + J1z*s_phi**2) * s_theta*c_theta*thetadot*psidot

        C11 = (J1y - J1z) * s_phi * c_phi * (thetadot**2 - c_theta**2 * psidot**2) + ((J1y - J1z) * (c_phi**2 - s_phi**2) - J1x) * (c_theta * thetadot * psidot)        
        C12 = 2 * (J1z - J1y) * s_phi * c_phi * phidot * thetadot + ((J1y - J1z) * (c_phi**2 - s_phi**2) + self.J1x) * c_theta * phidot * psidot - 0.5 * N33 * psidot**2
        C13 = c3_1 + c3_2 + c3_3 + c3_4
        # Return the C matrix
        return np.array([[C11],
                [C12],
                [C13],
                ])
        
    def _partialP(self, state: np.ndarray):
        # FIXME Fill in this function
        #extact any necessary variables from the state
        theta = state[1][0]
        # Return the partialP array
        return np.array([[0],
                        [(self.m1 * self.ell1 + self.m2*self.ell2)*P.g*cos(theta)],
                        [0],
                        ])
    
    def _tau(self, state: np.ndarray, force: float, torque: float):
        """
        Returns the tau matrix as defined in the hummingbird manual.

        Parameters
        ----------
        state : numpy.ndarray
            The state of the hummingbird. Contains phi, theta, psi, and their derivatives.
        force : float
            force = (fl + fr). e.g. the second element of the tau matrix becomes
            lT * force * cos(phi) using the above definition.
        torque : float
            torque = d(fl - fr). e.g. the first element of teh tau matrix just
            becomes torque, using the definition above.

        """
        # FIXME Fill in this function
        #extract any necessary variables from the state
        phi = state[0][0]
        theta = state[1][0]

        # Return the tau matrix
        return np.array([[torque],
                        [self.ellT * force * cos(phi)],
                        [self.ellT * force * cos(theta) * sin(phi) - torque * sin(theta)]])
    
    def _B(self):
        # FIXME Fill in this function
        # This needs no variables from the state
        
        # Return the B matrix
        return 0.001 * np.eye(3)


def saturate(u: np.ndarray, limit: float):
    for i in range(0, u.shape[0]):
        if abs(u[i][0]) > limit:
            u[i][0] = limit * np.sign(u[i][0])
    return u
