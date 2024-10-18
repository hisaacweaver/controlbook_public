import matplotlib.pyplot as plt
import numpy as np
import hummingbirdParam as P
from signalGenerator import SignalGenerator
from hummingbirdAnimation import HummingbirdAnimation
from dataPlotter import DataPlotter
from hummingbirdDynamics import HummingbirdDynamics
from ctrlEquilibrium import ctrlEquilibrium


# instantiate the hummingbird dyanmics
hummingbird = HummingbirdDynamics(alpha=0.0)

# instantiate the simulation plots and animation
dataPlot = DataPlotter()
animation = HummingbirdAnimation()
controler = ctrlEquilibrium()
t = P.t_start  # time starts at t_start
while t < P.t_end:  # main simulation loop

    # Propagate dynamics at rate Ts
    t_next_plot = t + P.t_plot
    while t < t_next_plot:
        ref = np.array([[0.], [0.], [0.]])
        pwm_right = 0.38
        pwm_left = 0.38
        pwm = controler.update(x=hummingbird.state)
        #u = np.array([[pwm], [pwm]])
        y = hummingbird.update(pwm)  # Propagate the dynamics
        t = t + P.Ts  # advance time by Ts

    # update animation and data plots at rate t_plot
    animation.update(t, hummingbird.state)
    dataPlot.update(t, hummingbird.state, pwm, ref)

    # the pause causes figure to be displayed during simulation
    plt.pause(0.05)

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()


#0.329 km value