
import matplotlib.pyplot as plt
import numpy as np
import blockbeamParam as P
from signalGenerator import signalGenerator
from blockbeamAnimation import blockbeamAnimation
from dataPlotter import dataPlotter
from blockbeamDynamics import blockbeamDynamics
from ctrlPD import ctrlPD

# instantiate blockbeam, controller, and reference classes
blockbeam = blockbeamDynamics(alpha=0.0)
controller = ctrlPD()
reference = signalGenerator(amplitude=.15, 
                            frequency=0.05, y_offset=0.25)

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = blockbeamAnimation()

t = P.t_start  # time starts at t_start
y = blockbeam.h()  # output of system at start of simulation

while t < P.t_end:  # main simulation loop
    # Get referenced inputs from signal generators
    # Propagate dynamics in between plot samples
    t_next_plot = t + P.t_plot

    # updates control and dynamics at faster simulation rate
    while t < t_next_plot: 
        r = reference.square(t)
        x = blockbeam.state
        u = controller.update(r, x)  # update controller
        y = blockbeam.update(u)  # propagate system
        t += P.Ts  # advance time by Ts

    # update animation and data plots
    animation.update(blockbeam.state)
    dataPlot.update(t, blockbeam.state, u, r)

    # the pause causes the figure to be displayed for simulation
    plt.pause(0.001)  

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
