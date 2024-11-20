import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button
import numpy as np
import Param as P
import matplotlib
matplotlib.use('tkagg')  # requires TkInter

def exit_program(event):
    exit()

class doublePendulumAnimation:
    def __init__(self):
        # Used to indicate initialization
        self.flagInit = True
        # Initializes a figure and axes object
        self.fig, self.ax = plt.subplots()        
        
        # Initializes a list object that will be used to
        # contain handles to the patches and line objects.
        self.handle = []
        self.length1 = P.ell
        self.length2 = P.ell
        self.width = P.width
        # Change the x,y axis limits
        plt.axis([-2.0*(self.length1 + self.length2), 2.0*(self.length1 + self.length2), 
                  -2.0*(self.length1 + self.length2), 2.0*(self.length1 + self.length2)])
        # Draw a base line
        plt.plot([0, self.length1], [0, 0],'k--')

        # Create exit button
        self.button_ax = plt.axes([0.8, 0.805, 0.1, 0.075])  # [left, bottom, width, height]
        self.exit_button = Button(self.button_ax, label='Exit', color='r',)
        self.exit_button.label.set_fontweight('bold')
        self.exit_button.label.set_fontsize(18)
        self.exit_button.on_clicked(exit_program)

    def update(self, x):
        # Process inputs to function
        theta1 = x[0][0]   # angle of first arm, rads
        theta2 = x[1][0]   # angle of second arm, rads
        X1 = [0, self.length1*np.cos(theta1)]  # X data points for first arm
        Y1 = [0, self.length1*np.sin(theta1)]  # Y data points for first arm
        X2 = [X1[1], X1[1] + self.length2*np.cos(theta2)]  # X data points for second arm
        Y2 = [Y1[1], Y1[1] + self.length2*np.sin(theta2)]  # Y data points for second arm

        # When the class is initialized, line objects will be
        # created and added to the axes. After initialization, the
        # line objects will only be updated.
        if self.flagInit == True:
            # Create the line objects and append their handles
            # to the handle list.
            line1, = self.ax.plot(X1, Y1, lw=5, c='blue')
            line2, = self.ax.plot(X2, Y2, lw=5, c='red')
            self.handle.append(line1)
            self.handle.append(line2)
            self.flagInit = False
        else:
            self.handle[0].set_xdata(X1)   # Update the first arm
            self.handle[0].set_ydata(Y1)
            self.handle[1].set_xdata(X2)   # Update the second arm
            self.handle[1].set_ydata(Y2)
        plt.draw()
