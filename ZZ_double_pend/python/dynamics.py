import numpy as np
import Param as P

class doublePendulumDynamics:
    def __init__(self, alpha=0.0):
        # Initial state conditions
        self.state = np.array([
            [P.theta1_0],      # initial angle of the first arm
            [P.theta2_0],      # initial angle of the second arm
            [P.theta1dot0],   # initial angular rate of the first arm
            [P.theta2dot0]    # initial angular rate of the second arm
        ])
        # Mass of the arms, kg
        self.m = P.m * (1. + alpha * (2. * np.random.rand() - 1.))
        # Length of the arms, m
        self.ell = P.ell * (1. + alpha * (2. * np.random.rand() - 1.))
        # Damping coefficient, Ns
        self.b = P.b * (1. + alpha * (2. * np.random.rand() - 1.))
        # the gravity constant is well known, so we don't change it.
        self.g = P.g
        # sample rate at which the dynamics are propagated
        self.Ts = P.Ts
        self.torque_limit = P.tau_max

    def update(self, u1, u2):
        # This is the external method that takes the inputs u1 and u2 at time
        # t and returns the output y at time t.
        # saturate the input torques
        u1 = saturate(u1, self.torque_limit)
        u2 = saturate(u2, self.torque_limit)
        self.rk4_step(u1, u2)  # propagate the state by one time sample
        y = self.h()  # return the corresponding output
        return y

    def f(self, state, tau1, tau2):
        # Return xdot = f(x,u), the system state update equations
        # re-label states for readability
        theta1 = state[0][0]
        theta2 = state[1][0]
        thetadot1 = state[2][0]
        thetadot2 = state[3][0]

        # Dynamics equations for double pendulum
        delta = theta2 - theta1
        den1 = (self.m * self.ell**2 * (2 - np.cos(delta)**2))
        den2 = (self.m * self.ell**2 * (2 - np.cos(delta)**2))

        thetaddot1 = (3 * self.g * np.sin(theta1) - self.b * thetadot1 + tau1 - self.m * self.ell * thetadot2**2 * np.sin(delta) * np.cos(delta) - 3 * self.g * np.sin(theta2) * np.cos(delta)) / den1
        thetaddot2 = (3 * self.g * np.sin(theta2) - self.b * thetadot2 + tau2 - self.m * self.ell * thetadot1**2 * np.sin(delta) * np.cos(delta) - 3 * self.g * np.sin(theta1) * np.cos(delta)) / den2

        xdot = np.array([[thetadot1], [thetadot2], [thetaddot1], [thetaddot2]])
        return xdot

    def h(self):
        # return the output equations
        # could also use input u if needed
        theta1 = self.state[0][0]
        theta2 = self.state[1][0]
        y = np.array([[theta1], [theta2]])
        return y

    def rk4_step(self, u1, u2):
        # Integrate ODE using Runge-Kutta RK4 algorithm
        F1 = self.f(self.state, u1, u2)
        F2 = self.f(self.state + self.Ts / 2 * F1, u1, u2)
        F3 = self.f(self.state + self.Ts / 2 * F2, u1, u2)
        F4 = self.f(self.state + self.Ts * F3, u1, u2)
        self.state = self.state + self.Ts / 6 * (F1 + 2 * F2 + 2 * F3 + F4)

def saturate(u, limit):
    if abs(u) > limit:
        u = limit * np.sign(u)
    return u
