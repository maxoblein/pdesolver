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

# u_T,diagnostics = Finite_Difference(initial_cond,[0,0],100,1000,(1.0,1.0,0.5),b_type = [1,0],plot = True)
# print(diagnostics[3])
error_plot_vary_mx('bd',initial_cond,[0,0],1000,(1.0,1.0,0.5),u_exact=u_exact)
