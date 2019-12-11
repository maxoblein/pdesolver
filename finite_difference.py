import numpy as np
import pylab as pl
from math import pi
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.integrate import trapz

def matrix_init(mx,diag_mul,sub_mul,b_type):
    '''
    Function tah defines the matrices used for each finite_difference
    method.

    Inputs: - mx number of gridpoints in space (int)
            - diag_mul the multiplier on the leading diagonal (float)
            - sub_mul the multiplier on the sub diagonal (float)
            - b_type the type of the boundary conditions (list)

    Outputs: - Mat the triadiagonal matrix (ndarray)
             - bc_vector the vector for boundary conditions (ndarray)
    '''
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
    '''
    Function that defines the update step used for each finite_difference
    method

    Inputs: - method of finite difference to be used (str)
            - A matrix to be solved on for each method (ndarray)
            - B second matrix only for crank (identity for others) (ndarray)
            - u_j vector of solutions at current timestep (ndarray)
            - bc_vector vector of boundary conditions at current timestep (ndarray)
    '''
    if method == 'fd':
        return A.dot(u_j) + bc_vector

    if method == 'bd':
        return spsolve(A,u_j + bc_vector)

    if method == 'cn':
        return spsolve(A,B.dot(u_j + bc_vector))


def solver(method,lmbda,mx,mt,deltat,deltax,u_j,bc,b_type,A,B,bc_vector):
    '''
    A function that implements the matrix form
     to solve pdes numerically.

    Inputs: -method: finite difference method to be used (str)
            -lmbda: The mesh fourier number for the system (float)
            -mx: The number of gridpoints in space (int)
            -mt: The number of gridpoints in time (int)
            -deltax: The distance between spacial gridpoints (float)
            -deltat: The distance between temporal gridpoints (float)
            -u_j: The initial condition vector U(x,0) (ndarray)
            -bc: The conditions at U(0,t) and U(L,t) (list)
            -b_type: The type of each boundary condition 0 for dirichlet 1 for neumann (list)
            -A: Matrix used to solve for next timestep (ndarray)
            -B: Matrix used to solve when using crank scheme (ndarray)
            -bc_vector: Vector to contain the boundary conditions of the system (ndarray)

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
    '''
    Function to compute the error with an exact solution

    Inputs: -u_T_approx the approx solution to the pde at t=T (ndarray)
            -u_T_exact the exact solution to the pde at t=T (ndarray)

    Output: -error between the true and approx solutions (float)
    '''
    error =  np.linalg.norm(u_T_exact - u_T_approx)
    error = error/np.size(u_T_exact)
    return error


def Finite_Difference(initial_cond,bc,mx,mt,params,method = 'cn',b_type = [0,0],u_exact = 0,plot = False):
    '''
    Function that implements the finite difference method
    for solving pdes numerically.

    Inputs: -initial_cond: Function for U(x,0) callable
            -bc: The boundary conditions at each end of domain ([int/callable,int/callable])
            -mx: Number of gridpoints in space (float)
            -mt: Number of gripoints in time (float)
            -params: parameters of the system [kappa,Length of domain in x,Time to solve to]
            -method: finite finite_difference method to use (default = 'cn' but 'fd' for forward and 'bd' for backward)
            -b_type: type of the boundary conditions 0 for dirichlet 1 for neumann (list, default = [0,0])
            -u_exact: Optional function for the exact solution U(x,T) callable
            -plot: Optional boolean to display plot of numerical solution wit true sol overlayed

    Outputs: -u_T: solutiona at end of time domain.
             -diagnostics: information useful for errors [deltax,deltat,lmbda,error between solution and one with 2 time the spacial and temporal gridpoints]
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

    if method not in ('fd' ,'bd' , 'cn'):
        print('Unsupported finite diference scheme')
        return [1,1]

    #check that program will run
    if method == 'fd' and lmbda > 0.5:
        print('Error forward difference is only stable if lambda is less than 0.5')
        return [1,1]

    # set up the solution variables
    u_j = initial_cond(x,params)

    if method == 'fd':
        # Define diagonal matrix for forwards
        A,bc_vector = matrix_init(mx,(1-2*lmbda),lmbda,b_type)
        B = None

    if method == 'bd':
        # Define diagonal matrix for backwards
        A,bc_vector = matrix_init(mx,(1+2*lmbda),-lmbda,b_type)
        B = None

    if method == 'cn':
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
