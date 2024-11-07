
import matplotlib.pyplot as plt
import blockbeamParam as P
from signalGenerator import signalGenerator
from blockbeamAnimation import blockbeamAnimation
from dataPlotter import dataPlotter
from blockbeamDynamics import blockbeamDynamics
from ctrlStateFeedback import ctrlStateFeedback

# instantiate blockbeam, controller, and reference classes
blockbeam = blockbeamDynamics()
controller = ctrlStateFeedback()
reference = signalGenerator(amplitude=0.5, frequency=0.04)

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = blockbeamAnimation()

t = P.t_start  # time starts at t_start
y = blockbeam.h()  # output of system at start of simulation

while t < P.t_end/5:  # main simulation loop
    # Propagate dynamics in between plot samples
    t_next_plot = t + P.t_plot

    # updates control and dynamics at faster simulation rate
    while t < t_next_plot: 
        r = reference.square(t)  # reference input
        x = blockbeam.state
        u = controller.update(P.length/2.0, x)  # update controller
        y = blockbeam.update(u)  # propagate system
        t += P.Ts  # advance time by Ts
    # update animation and data plots
    animation.update(blockbeam.state)
    dataPlot.update(t, blockbeam.state, u, r)
    plt.pause(0.0001)  

# Keeps the program from closing until user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
