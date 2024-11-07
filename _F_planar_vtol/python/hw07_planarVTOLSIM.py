import matplotlib.pyplot as plt
import numpy as np
import VTOLParam as P
from signalGenerator import signalGenerator
from VTOLAnimation import VTOLAnimation
from dataPlotter import dataPlotter
from VTOLDynamics import VTOLDynamics
from ctrlPD import ctrlPD

VTOL = VTOLDynamics()

# instantiate reference input classes, these are not actual values, 
# just values to allow us to plot 
z_reference = signalGenerator(amplitude=2.5, frequency=0.04, y_offset=3.0)
h_reference = signalGenerator(amplitude=3.0, frequency=0.03, y_offset=5.0)

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = VTOLAnimation()
controller = ctrlPD()

t = P.t_start  # time starts at t_start
while t < P.t_end:  # main simulation loop
    # set variables
    t_next_plot = t + P.t_plot

    while t < t_next_plot:
        z_ref = z_reference.square(t)
        h_ref = h_reference.square(t)

        r = np.array([[z_ref], [h_ref]])
        d = np.array([[0.0],[0.0]])
        x = VTOL.state
        u = controller.update(r,x)
        # F = P.Fe * (0.1134 * (-VTOL.state[1][0]- h_ref))
        # self.kp_h * (h_r - h) - self.kd_h * hdot
        # u = np.array([[F], [F]])
        y = VTOL.update(u + d)
        t = t + P.Ts

    animation.update(VTOL.state, z_ref)
    dataPlot.update(t, VTOL.state, u, z_ref, h_ref)

    # advance time by t_plot
    plt.pause(0.001)  # allow time for animation to draw

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
