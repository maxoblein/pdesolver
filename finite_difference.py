import numpy as np
import pylab as pl
from math import pi
from scipy import sparse
from scipy.sparse.linalg import spsolve

def forward(lmbda,u_j,u_jp1):
    # Define diagonal matrix for forwards
    upper = (lmbda) * np.ones(mx-2)
    lower = upper
    diag = (1-2*lmbda) * np.ones(mx-1)
    A = sparse.diags([upper,diag,lower],[1,0,-1],format = 'csc')

    for n in range(1, mt+1):
        # solve up to end of time period
        u_jp1[1:-1] = A.dot(u_j[1:-1])

        #apply boundary cond
        u_jp1[0] = boundary_conds[0]; u_jp1[mx] = boundary_conds[1]

        #update
        u_j[:] = u_jp1[:]

    return u_j

def backward(lmbda,u_j,u_jp1):
    # Define diagonal matrix for backwards
    upper = (-lmbda) * np.ones(mx-2)
    lower = upper
    diag = (1+2*lmbda) * np.ones(mx-1)
    A = sparse.diags([upper,diag,lower],[1,0,-1],format = 'csc')

    for n in range(1, mt+1):
        # solve up to end of time period
        u_jp1[1:-1] = spsolve(A,u_j[1:-1])

        #apply boundary cond
        u_jp1[0] = boundary_conds[0]; u_jp1[mx] = boundary_conds[1]

        #update
        u_j[:] = u_jp1[:]

    return u_j

def central(lmbda,u_j,u_jp1):
    upper = (lmda/2) * np.ones(mx-2)
    lower = upper
    Adiag = (1+lmbda) * np.ones(mx-1)
    Bdiag = (1-lmbda) * np.ones(mx-1)
    A = sparse.diags([-1*upper,Adiag,-1*lower],[1,0,-1],format = 'csc')
    B = sparse.diags([upper,Bdiag,lower],[1,0,-1],format = 'csc')

    for n in range(1, mt+1):
        # solve up to end of time period
        u_jp1[1:-1] = spsolve(A,B.dot(u_j[1:-1]))

        #apply boundary cond
        u_jp1[0] = boundary_conds[0]; u_jp1[mx] = boundary_conds[1]

        #update
        u_j[:] = u_jp1[:]

    return u_j

def Finite_Difference(method,intial_cond,boundary_conds,mx,mt,params,u_exact = 0):
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

    # set up the solution variables
    u_j = np.zeros(x.size)        # u at current time step
    u_jp1 = np.zeros(x.size)      # u at next time step

    # Set initial condition
    for i in range(0, mx+1):
        u_j[i] = intial_cond(x[i])

    if method == 'forward':
        # Define diagonal matrix for forwards
        u_T = forward(lmbda,u_j,u_jp1,boundary_conds)

    if method == 'backward':
        # Define diagonal matrix for backwards
        u_T = backward(lmbda,u_j,u_jp1,boundary_conds)

    if method == 'crank-nicholson':
        # define diagonal matrices for crank nicholson
        u_T = central(lmbda,u_j,u_jp1,boundary_conds)

    pl.plot(x,u_j,'ro',label='num')
    xx = np.linspace(0,L,250)
    if u_exact != 0:
        pl.plot(xx,u_exact(xx,T),'b-',label='exact')
    pl.xlabel('x')
    pl.ylabel('u(x,0.5)')
    pl.legend(loc='upper right')
    pl.show()
