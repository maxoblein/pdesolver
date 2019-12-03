import numpy as np
import pylab as pl
from math import pi
from scipy import sparse
from scipy.sparse.linalg import spsolve

def forward(lmbda,mx,mt,u_j,boundary_conds):
    '''
    A function that implements the matrix form of the forward difference
    method to solve pdes numerically.

    Inputs: -lmbda: The mesh fourier number for the system (float)
            -mx: The number of gridpoints in space (float)
            -mt: The number of gridpoints in time (float)
            -u_j: The initial condition vector U(x,0) (ndarray)
            -boundary_conds: The conditions at U(0,t) and U(L,t)

    Output: -u_T: The solution to the pde at U(x,T) (ndarray)
    '''
    # Define diagonal matrix for forwards
    upper = (lmbda) * np.ones(mx-2)
    lower = upper
    diag = (1-2*lmbda) * np.ones(mx-1)
    A = sparse.diags([upper,diag,lower],[1,0,-1],format = 'csc')
    u_jp1 = np.zeros(u_j.size)      # u at next time step

    for n in range(1, mt+1):
        # solve up to end of time period
        u_jp1[1:-1] = A.dot(u_j[1:-1])

        #apply boundary cond
        u_jp1[0] = boundary_conds[0]; u_jp1[mx] = boundary_conds[1]

        #update
        u_j[:] = u_jp1[:]
    u_T = u_j
    return u_T

def backward(lmbda,mx,mt,u_j,boundary_conds):
    '''
    A function that implements the matrix form of the backward difference
    method to solve pdes numerically.

    Inputs: -lmbda: The mesh fourier number for the system (float)
            -mx: The number of gridpoints in space (float)
            -mt: The number of gridpoints in time (float)
            -u_j: The initial condition vector U(x,0) (ndarray)
            -boundary_conds: The conditions at U(0,t) and U(L,t)

    Output: -u_T: The solution to the pde at U(x,T) (ndarray)
    '''
    # Define diagonal matrix for backwards
    upper = (-lmbda) * np.ones(mx-2)
    lower = upper
    diag = (1+2*lmbda) * np.ones(mx-1)
    A = sparse.diags([upper,diag,lower],[1,0,-1],format = 'csc')
    u_jp1 = np.zeros(u_j.size)      # u at next time step

    for n in range(1, mt+1):
        # solve up to end of time period
        u_jp1[1:-1] = spsolve(A,u_j[1:-1])

        #apply boundary cond
        u_jp1[0] = boundary_conds[0]; u_jp1[mx] = boundary_conds[1]

        #update
        u_j[:] = u_jp1[:]

    u_T = u_j
    return u_T

def central(lmbda,mx,mt,u_j,boundary_conds):
    '''
    A function that implements the matrix form of the Crank-Nicholson
    method to solve pdes numerically.

    Inputs: -lmbda: The mesh fourier number for the system (float)
            -mx: The number of gridpoints in space (float)
            -mt: The number of gridpoints in time (float)
            -u_j: The initial condition vector U(x,0) (ndarray)
            -boundary_conds: The conditions at U(0,t) and U(L,t)

    Output: -u_T: The solution to the pde at U(x,T) (ndarray)
    '''
    upper = (lmbda/2) * np.ones(mx-2)
    lower = upper
    Adiag = (1+lmbda) * np.ones(mx-1)
    Bdiag = (1-lmbda) * np.ones(mx-1)
    A = sparse.diags([-1*upper,Adiag,-1*lower],[1,0,-1],format = 'csc')
    B = sparse.diags([upper,Bdiag,lower],[1,0,-1],format = 'csc')
    u_jp1 = np.zeros(u_j.size)      # u at next time step

    for n in range(1, mt+1):
        # solve up to end of time period
        u_jp1[1:-1] = spsolve(A,B.dot(u_j[1:-1]))

        #apply boundary cond
        u_jp1[0] = boundary_conds[0]; u_jp1[mx] = boundary_conds[1]

        #update
        u_j[:] = u_jp1[:]

    u_T = u_j
    return u_T

def find_error_with_true(u_T_approx,u_T_exact):
    return np.sqrt(np.sum((u_T_exact - u_T_approx)**2))


def Finite_Difference(method,initial_cond,boundary_conds,mx,mt,params,u_exact = 0,plot = False):
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
    print("deltax=",deltax)
    print("deltat=",deltat)
    print("lambda=",lmbda)

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
        u_T = forward(lmbda,mx,mt,u_j,boundary_conds)

    if method == 'backward':
        # Define diagonal matrix for backwards
        u_T = backward(lmbda,mx,mt,u_j,boundary_conds)

    if method == 'crank':
        # define diagonal matrices for crank nicholson
        u_T = central(lmbda,mx,mt,u_j,boundary_conds)

    if u_exact != 0:
        error = find_error_with_true(u_T,u_exact(x,T,params))

    if u_exact == 0:
        error = 0

    if plot == True:
        pl.plot(x,u_j,'ro',label='num')
        xx = np.linspace(0,L,250)
        if u_exact != 0:
            pl.plot(xx,u_exact(xx,T,params),'b-',label='exact')
        pl.xlabel('x')
        pl.ylabel('u(x,0.5)')
        pl.legend(loc='upper right')
        pl.show()

    diagnostics = [error,deltax,deltat,lmbda]
    return u_T,diagnostics

def error_plot_vary_mt(method,initial_cond,boundary_conds,mx,params,u_exact = 0):
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
        for n in range(1,12):
            mt = 2**n
            u_T,diagnostics = Finite_Difference(method,initial_cond,boundary_conds,mx,mt,params,u_exact = u_exact)
            deltat_list.append(diagnostics[2])
            error_list.append(diagnostics[0])
        pl.loglog(deltat_list,error_list)


    else:
        u_T_list = []
        for n in range(1,12):
            mt = 2**n
            u_T,diagnostics = Finite_Difference(method,initial_cond,boundary_conds,mx,mt,params,u_exact = u_exact)
            u_T_list.append(u_T)
            deltat_list.append(diagnostics[2])
        for i in range(len(u_T_list)-1):
            error_list.append(np.sqrt(np.sum((u_T_list[i+1] - u_T_list[i])**2)))
        pl.loglog(deltat_list[:-1],error_list)
    pl.xlabel('Number of gridpoints in time')
    pl.ylabel('Error between finite difference and exact solution')
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
        for n in range(1,12):
            mx = 2**n
            u_T,diagnostics = Finite_Difference(method,initial_cond,boundary_conds,mx,mt,params,u_exact = u_exact)
            deltax_list.append(diagnostics[1])
            error_list.append(diagnostics[0])
        pl.loglog(deltax_list,error_list)


    else:
        u_T_list = []
        for n in range(1,12):
            mx = 2**n
            u_T,diagnostics = Finite_Difference(method,initial_cond,boundary_conds,mx,mt,params,u_exact = u_exact)
            u_T_list.append(u_T)
            deltax_list.append(diagnostics[1])
        for i in range(len(u_T_list)-1):
            len_now = len(u_T_list[i])
            len_next = len(u_T_list[i+1])
            error_list.append(np.sqrt((u_T_list[i+1][int(np.round(len_next/2))])-(u_T_list[i][int(np.round(len_now/2))])**2))
        pl.loglog(deltax_list[:-1],error_list)
    pl.xlabel('Number of gridpoints in space')
    pl.ylabel('Error between finite difference and exact solution')
    pl.show()
    return True
