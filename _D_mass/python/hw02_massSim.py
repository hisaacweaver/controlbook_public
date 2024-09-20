import matplotlib.pyplot as plt
import numpy as np
import massParam as P
from signalGenerator import signalGenerator
from massAnimation import massAnimation
from dataPlotter import dataPlotter

# instantiate reference input classes, these are not actual values, 
# just values to allow us to plot 
z_plot = signalGenerator(amplitude=2.0*np.pi, frequency=0.1)
f_plot = signalGenerator(amplitude=2, frequency=.5)

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = massAnimation()

t = P.t_start  # time starts at t_start
while t < P.t_end:  # main simulation loop
    # set variables
    z = z_plot.sin(t)
    f = f_plot.sawtooth(t)

    # update animation
    z_dot = 0.0
    state = np.array([[z], [z_dot]])  #state is made of theta, and theta_dot
    animation.update(state)
    dataPlot.update(t, state, f)

    # advance time by t_plot
    t += P.t_plot  
    plt.pause(0.001)  # allow time for animation to draw

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
