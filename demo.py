from finite_difference import *

def initial_cond(x):
    # initial temperature distribution
    y = np.sin(pi*x/L)
    return y
def u_exact(x,t):
    # the exact solution
    y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
    return y

Finite_Difference(forward,initial_cond,[0,0],40,1000,[1.0,1.0,0.5],u_exact)
