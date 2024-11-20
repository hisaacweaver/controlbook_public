import numpy as np
import sympy
from sympy import *
from sympy.physics.vector.printing import vlatex
from IPython.display import Math, display

init_printing()

def dotprint(expr):
    display(Math(vlatex(expr)))

def derive_Lagrangian(L, q, q_dot):
    term_1 = (sympy.tensor.derive_by_array(L, q_dot)).diff(t)
    term_2 = sympy.tensor.derive_by_array(L, q)
    return term_1 - term_2


t = symbols('t')
theta1, theta2 = symbols(r'\theta_1, \theta_2', cls=Function)
theta1 = theta1(t)
theta2 = theta2(t)

dotprint(theta1)

