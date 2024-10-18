import matplotlib.pyplot as plt
import numpy as np
import hummingbirdParam as P
from signalGenerator import SignalGenerator
from hummingbirdAnimation import HummingbirdAnimation
from hummingbirdDynamics import HummingbirdDynamics
from dataPlotter import DataPlotter

# instantiate reference input classes
torque1 = SignalGenerator(amplitude=1.5, frequency=0.05)
torque2 = SignalGenerator(amplitude=0.5, frequency=0.05)

# instantiate the simulation plots and animation
dataPlot = DataPlotter()
animation = HummingbirdAnimation()
dynamics  = HummingbirdDynamics()

t = P.t_start  # time starts at t_start
while t < P.t_end:  # main simulation loop

    ref = np.array([[0], [0], [0]])
    force = 0
    torque = np.array([[torque1.sin(t)],[torque2.sin(t)]])
    dynamics.update(torque)
    state = dynamics.state
    animation.update(t, state)
    dataPlot.update(t, state, torque, ref)

    t = t + P.t_plot  # advance time by t_plot
    plt.pause(0.05)

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()