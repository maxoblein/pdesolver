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

u_T,diagnostics = Finite_Difference('crank',initial_cond,[0,0],10,1000,(1.0,1.0,0.5),u_exact = u_exact)

deltat_list = []
error_list = []
for n in range(1,10):
    mt = 2**n
    u_T,diagnostics = Finite_Difference('backward',initial_cond,[0,0],10,mt,(1.0,1.0,0.5),u_exact = u_exact)
    deltat_list.append(diagnostics[2])
    error_list.append(diagnostics[0])
pl.loglog(deltat_list,error_list)
pl.xlabel('Number of gridpoints in space')
pl.ylabel('Error between finite difference and exact solution')
pl.show()
