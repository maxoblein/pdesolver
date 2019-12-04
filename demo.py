from finite_difference import *

def initial_cond(x,params):
    # initial temperature distribution
    L = params[1]
    y = np.sin(pi*x/L)
    return y
def u_exact(x,t,params):
    # the exact solution
    kappa = params[0]
    L = params[1]

    y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
    return y

def lhs(t):
    return t**2

def rhs(t):
    return -2*t

u_T,diagnostics = Finite_Difference('backward',initial_cond,[lhs,rhs],10,1000,(1.0,1.0,0.5),plot=True)

#error_plot_vary_mt('backward',initial_cond,[0,0],10,(1.0,1.0,0.5))
