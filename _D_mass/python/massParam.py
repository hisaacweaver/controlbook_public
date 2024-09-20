# mass-spring-damper Parameter File
import numpy as np

# Physical parameters of the arm known to the controller
m = 1 # mass kg
k = 1 # spring constant Kg/s^2
b = 1  # damping coefficient Kg/s

# parameters for animation
length = 5.0
width = 1.0

# Initial Conditions
z0 = 0  # initial position of mass, m
zdot0 = 0 # initial velocity of mass m/s

# Simulation Parameters
t_start = 0.0  # Start time of simulation
t_end = 50.0  # End time of simulation
Ts = 0.01  # sample time for simulation
t_plot = 0.1  # the plotting and animation is updated at this rate  

# dirty derivative parameters
# sigma =  # cutoff freq for dirty derivative

# saturation limits
F_max = 1  # Max force, N

