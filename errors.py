from finite_difference import *

def error_plot_vary_mt(method,initial_cond,bc,mx,params,u_exact = 0):
    '''
    Function to plot a loglog error plot given a static mx and varying mt

    Inputs: -method: e.g. 'forward' for forward difference (string)
            -initial_cond: Function for U(x,0) callable
            -boundary_conds:
            -mx: Number of gridpoints in space (float)
            -params: parameters of the system [kappa,Length of domain in x,Time to solve to]
            -u_exact: Optional function for the exact solution U(x,T) callable
    '''
    deltat_list = []
    error_list = []
    if u_exact != 0:
        for n in range(3,15):
            mt = 2**n
            u_T,diagnostics = Finite_Difference(initial_cond,bc,mx,mt,params,method = method,b_type = [0,0])
            u_T_exact = u_exact(np.linspace(0, params[1], mx+1),params[2],params)
            deltat_list.append(diagnostics[1])
            error_list.append(find_error_with_true(u_T,u_T_exact,diagnostics[0]))
        pl.loglog(deltat_list,error_list)
        slope, intercept = np.polyfit(np.log(deltat_list), np.log(error_list), 1)
        print(slope)


    else:
        u_T_list = []
        for n in range(3,15):
            mt = 2**n
            u_T,diagnostics = Finite_Difference(initial_cond,bc,mx,mt,params,method,b_type = [0,0])
            u_T_list.append(u_T)
            deltat_list.append(diagnostics[2])
        for i in range(len(u_T_list)-1):
            error_list.append(find_error_with_true(u_T_list[i],u_T_list[i+1],diagnostics[0]))

        pl.loglog(deltat_list[:-1],error_list)
        slope, intercept = np.polyfit(np.log(deltat_list[:-1]), np.log(error_list), 1)
        print(slope)
    pl.xlabel(r'$\Delta t$')
    pl.ylabel(r'Error')
    pl.show()
    return True

def error_plot_vary_mx(method,initial_cond,boundary_conds,mt,params,u_exact = 0):
    '''
    Function to plot a loglog error plot given a static mt and varying mx

    Inputs: -method: e.g. 'forward' for forward difference (string)
            -initial_cond: Function for U(x,0) callable
            -boundary_conds:
            -mt: Number of gridpoints in time (float)
            -params: parameters of the system [kappa,Length of domain in x,Time to solve to]
            -u_exact: Optional function for the exact solution U(x,T) callable
    '''
    deltax_list = []
    error_list = []
    if u_exact != 0:
        for n in range(3,15):
            mx = 2**n
            u_T,diagnostics = Finite_Difference(initial_cond,boundary_conds,mx,mt,params,method = method,b_type = [0,0],u_exact = u_exact)
            u_T_exact = u_exact(np.linspace(0, params[1], mx+1),params[2],params)
            deltax_list.append(diagnostics[0])
            error = abs(trapz(u_T_exact,dx=diagnostics[0]) - trapz(u_T,dx=diagnostics[0]))
            error_list.append(error)
        slope, intercept = np.polyfit(np.log(deltax_list), np.log(error_list), 1)
        print(slope)

        pl.loglog(deltax_list,error_list)


    else:
        u_T_list = []
        for n in range(3,15):
            mx = 2**n
            u_T,diagnostics = Finite_Difference(initial_cond,boundary_conds,mx,mt,params,method = method,b_type = [0,0],u_exact = u_exact)
            u_T_list.append(u_T)
            deltax_list.append(diagnostics[0])
        for i in range(len(u_T_list)-1):

            error = abs(trapz(u_T_list[i+1],dx = deltax_list[i+1]) - trapz(u_T_list[i],dx = deltax_list[i]))
            error_list.append(error)

        slope, intercept = np.polyfit(np.log(deltax_list[:-1]), np.log(error_list), 1)
        print(slope)
        pl.loglog(deltax_list[:-1],error_list)
    pl.title('Plot of error trends in Crank-Nicolson')
    pl.xlabel(r'$\Delta x$')
    pl.ylabel('Error')
    pl.show()
    return True
