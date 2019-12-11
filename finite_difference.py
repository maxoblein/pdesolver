import numpy as np
import pylab as pl
from math import pi
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.integrate import trapz

def matrix_init(mx,diag_mul,sub_mul,b_type):
    if b_type[0] == 0 and b_type[1] == 0:
        size = mx-1
        multiplier = [1,1]
    if b_type[0] == 1 and b_type[1] == 1:
        size = mx+1
        multiplier = [2,2]
    if b_type[0] != b_type[1]:
        size = mx
        if b_type[0] == 1:
            multiplier = [2,1]
        else:
            multiplier = [1,2]

    diag = diag_mul * np.ones(size)
    sub_diag = sub_mul * np.ones(size-1)
    sup_diag = np.copy(sub_diag)
    sup_diag[0] = sup_diag[0] * multiplier[0]
    sub_diag[-1] = sub_diag[-1] * multiplier[1]
    Mat = sparse.diags([sup_diag,diag,sub_diag],[1,0,-1],format = 'csc')
    bc_vector = np.zeros(size)
    return Mat, bc_vector

def update_step(method,A,B,u_j,bc_vector):
    if method == 'forward':
        return A.dot(u_j) + bc_vector

    if method == 'backward':
        return spsolve(A,u_j + bc_vector)

    if method == 'crank':
        return spsolve(A,B.dot(u_j + bc_vector))


def solver(method,lmbda,mx,mt,deltat,deltax,u_j,bc,b_type,A,B,bc_vector):
    '''
    A function that implements the matrix form
     to solve pdes numerically.

    Inputs: -method: finite difference method to be used
            -lmbda: The mesh fourier number for the system (float)
            -mx: The number of gridpoints in space (float)
            -mt: The number of gridpoints in time (float)
            -u_j: The initial condition vector U(x,0) (ndarray)
            -bc: The conditions at U(0,t) and U(L,t)

    Output: -u_T: The solution to the pde at U(x,T) (ndarray)
    '''
    u_jp1 = np.zeros(u_j.size)
    for n in range(1,mt+1):
        if b_type == [0,0]:
            bc_vector[0] = lmbda * bc[0](n*deltat)
            bc_vector[-1] = lmbda * bc[1](n*deltat)
            u_jp1[1:-1] = update_step(method,A,B,u_j[1:-1],bc_vector)
            u_jp1[0] = bc[0](n*deltat); u_jp1[-1] = bc[1](n*deltat)

        if b_type == [1,1]:
            bc_vector[0] = 2*lmbda*deltax * bc[0](n*deltat)
            bc_vector[-1] = 2*lmbda*deltax * bc[1](n*deltat)
            u_jp1 = update_step(method,A,B,u_j,bc_vector)

        if b_type == [0,1]:
            bc_vector[0] = lmbda * bc[0](n*deltat)
            bc_vector[-1] = 2*lmbda*deltax * bc[1](n*deltat)
            u_jp1[1:] = update_step(method,A,B,u_j[1:],bc_vector)
            u_jp1[0] = bc[0](n*deltat)

        if b_type == [1,0]:
            bc_vector[0] = 2*lmbda*deltax * bc[0](n*deltat)
            bc_vector[-1] = lmbda* bc[1](n*deltat)
            u_jp1[:-1] = update_step(method,A,B,u_j[:-1],bc_vector)
            u_jp1[-1] = bc[1](n*deltat)

        u_j[:] = u_jp1[:]

    u_T = u_j
    return u_T

def find_error_with_true(u_T_approx,u_T_exact):
    #return np.sqrt(np.sum((u_T_exact - u_T_approx)**2))
    error =  np.linalg.norm(u_T_exact - u_T_approx)
    error = error/np.size(u_T_exact)
    return error


def Finite_Difference(method,initial_cond,bc,mx,mt,params,b_type = [0,0],u_exact = 0,plot = False):
    '''
    Function that implements the finite difference method
    for solving pdes numerically.

    Inputs: -method: e.g. 'forward' for forward difference (string)
            -initial_cond: Function for U(x,0) callable
            -boundary_conds:
            -mx: Number of gridpoints in space (float)
            -mt: Number of gripoints in time (float)
            -params: parameters of the system [kappa,Length of domain in x,Time to solve to]
            -u_exact: Optional function for the exact solution U(x,T) callable
            -plot: Optional boolean to display plot of numerical solution wit true sol overlayed
    '''
    # set up the numerical environment variables
    kappa = params[0]
    L = params[1]
    T = params[2]
    x = np.linspace(0, L, mx+1)     # mesh points in space
    t = np.linspace(0, T, mt+1)     # mesh points in time
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    lmbda = kappa*deltat/(deltax**2)    # mesh fourier number
    # print("deltax=",deltax)
    # print("deltat=",deltat)
    # print("lambda=",lmbda)

    for i in range(len(bc)):
        if isinstance(bc[i],float) or isinstance(bc[i],int):
            val = bc[i]
            bc[i] = lambda t : val

    if (b_type[0] != 0  and b_type[0] != 1) or (b_type[1] != 0  and b_type[1] != 1):
        print('Incorrect boundary condition types')
        return [1,1]


    #check that program will run
    if method == 'forward' and lmbda > 0.5:
        print('Error forward difference is only stable if lambda is less than 0.5')
        return [1,1]

    # set up the solution variables
    u_j = np.zeros(x.size)        # u at current time step


    # Set initial condition
    for i in range(0, mx+1):
        u_j[i] = initial_cond(x[i],params)

    if method == 'forward':
        # Define diagonal matrix for forwards
        A,bc_vector = matrix_init(mx,(1-2*lmbda),lmbda,b_type)
        B = None

    if method == 'backward':
        # Define diagonal matrix for backwards
        A,bc_vector = matrix_init(mx,(1+2*lmbda),-lmbda,b_type)
        B = None

    if method == 'crank':
        A,bc_vector = matrix_init(mx,(1+lmbda),(-lmbda/2),b_type)
        B,bc_vector = matrix_init(mx,(1-lmbda),(lmbda/2),b_type)
        # define diagonal matrices for crank nicholson

    u_T = solver(method,lmbda,mx,mt,deltat,deltax,u_j,bc,b_type,A,B,bc_vector)

    if plot == True:
        pl.plot(x,u_j,'ro',label='num')
        xx = np.linspace(0,L,250)
        if u_exact != 0:
            pl.plot(xx,u_exact(xx,T,params),'b-',label='exact')
        pl.xlabel('x')
        pl.ylabel('u(x,0.5)')
        pl.title(r'Temperature distribution at $t = T$')
        #pl.legend(loc='upper right')
        pl.show()

    diagnostics = [deltax,deltat,lmbda]
    return u_T,diagnostics

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
            u_T,diagnostics = Finite_Difference(method,initial_cond,bc,mx,mt,params,b_type = [0,0])

            u_T_exact = u_exact(np.linspace(0, params[1], mx+1),params[2],params)
            deltat_list.append(diagnostics[1])
            error_list.append(find_error_with_true(u_T,u_T_exact))
        pl.loglog(deltat_list,error_list)
        slope, intercept = np.polyfit(np.log(deltat_list), np.log(error_list), 1)
        print(slope)
        pl.title('Plot of error trends in Crank-Nicolson')


    else:
        u_T_list = []
        for n in range(3,15):
            mt = 2**n
            u_T,diagnostics = Finite_Difference(method,initial_cond,bc,mx,mt,params,b_type = [0,0])
            u_T_list.append(u_T)
            deltat_list.append(diagnostics[2])
        for i in range(len(u_T_list)-1):
            error_list.append(np.sqrt(np.sum((u_T_list[i+1] - u_T_list[i])**2)))

        pl.loglog(deltat_list[:-1],error_list)
        slope, intercept = np.polyfit(np.log(deltat_list[:-1]), np.log(error_list), 1)
        print(slope)
        pl.title('Plot of error trends in Crank-Nicolson')
    pl.xlabel(r'$\Delta t$')
    pl.ylabel(r'Error between finite difference at $mt$ and $2mt$')
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
            u_T,diagnostics = Finite_Difference(method,initial_cond,boundary_conds,mx,mt,params,b_type = [0,0],u_exact = u_exact)
            u_T_exact = u_exact(np.linspace(0, params[1], mx+1),params[2],params)
            deltax_list.append(diagnostics[0])
            error = abs(trapz(u_T_exact,dx=diagnostics[0]) - trapz(u_T,dx=diagnostics[0]))
            error_list.append(error)
            print(find_error_with_true(u_T,u_T_exact))
        slope, intercept = np.polyfit(np.log(deltax_list), np.log(error_list), 1)
        print(slope)

        pl.loglog(deltax_list,error_list)


    else:
        u_T_list = []
        for n in range(3,15):
            mx = 2**n
            u_T,diagnostics = Finite_Difference(method,initial_cond,boundary_conds,mx,mt,params,b_type = [0,0],u_exact = u_exact)
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
    pl.ylabel('Error between finite difference and exact solution')
    pl.show()
    return True
