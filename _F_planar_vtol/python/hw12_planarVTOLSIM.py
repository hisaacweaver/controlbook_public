import matplotlib.pyplot as plt
import numpy as np
import VTOLParam as P
from signalGenerator import signalGenerator
from VTOLAnimation import VTOLAnimation
from dataPlotter import dataPlotter
from VTOLDynamics import VTOLDynamics
from ctrlStateFeedbackIntegrator import ctrlStateFeedbackIntegrator as ctrlStateFeedback

# instantiate VTOL, controller, and reference classes
VTOL = VTOLDynamics()
controller = ctrlStateFeedback()
h_reference = signalGenerator(amplitude=3.0, frequency=0.03, y_offset=5.0)
z_reference = signalGenerator(amplitude=4.0, frequency=0.02, y_offset=5.0)

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = VTOLAnimation()

t = P.t_start  # time starts at t_start
y = VTOL.h()  # output of system at start of simulation
x = VTOL.state
while t < P.t_end:  # main simulation loop
    # Propagate dynamics in between plot samples
    t_next_plot = t + P.t_plot
    while t < t_next_plot:  # updates control and dynamics at faster simulation rate
        h_r = h_reference.square(t)  # reference input
        z_r = z_reference.square(t)
        x = VTOL.state
        u = controller.update(z_r, h_r, x) # update controller
        y = VTOL.update(u)  # propagate system
        t += P.Ts  # advance time by Ts
    # update animation and data plots
    animation.update(VTOL.state)
    dataPlot.update(t, VTOL.state, u, h_r, z_r)
    plt.pause(0.01)  # the pause causes the figure to be displayed during the simulation

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
