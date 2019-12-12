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

def test_forward_unstable():
    mx = 40
    mt = 1000
    u_T = Finite_Difference(initial_cond,[0,0],40,1000,(1.0,1.0,0.5),method = 'fd',b_type = [0,0])
    if u_T == [1,1]:
        print('Forward unstable test passed')
    else:
        print('Forward unstable test failed')

def test_forward_output():
    tol = 1e-3
    u_T,diagnostics = Finite_Difference(initial_cond,[0,0],10,1000,(1.0,1.0,0.5),method='fd',b_type = [0,0])
    x = np.linspace(0,1,11)
    u_e = u_exact(x,0.5,(1.0,1.0,0.5))
    error = abs(np.sum(u_e - u_T)/11)
    if error < tol:
        print('Forward output test passed')
    else:
        print('Forward output test failed')

def test_backward_output():
    tol = 1e-4
    u_T,diagnostics = Finite_Difference(initial_cond,[0,0],40,1000,(1.0,1.0,0.5),method = 'bd',b_type = [0,0])
    x = np.linspace(0,1,41)
    u_e = u_exact(x,0.5,(1.0,1.0,0.5))
    error = abs(np.sum(u_e - u_T)/41)
    if error < tol:
        print('Backward output test passed')
    else:
        print('Backward output test failed')

def test_crank_output():
    tol = 1e-4
    u_T,diagnostics = Finite_Difference(initial_cond,[0,0],40,1000,(1.0,1.0,0.5),method = 'cn',b_type = [0,0])
    x = np.linspace(0,1,41)
    u_e = u_exact(x,0.5,(1.0,1.0,0.5))
    error = abs(np.sum(u_e - u_T)/41)
    if error < tol:
        print('Crank-Nicolson output test passed')
    else:
        print('Crank-Nicolson output test failed')

def test_incorrect_b_type():
    u_T_forward = Finite_Difference(initial_cond,[0,0],10,1000,(1.0,1.0,0.5),method = 'fd',b_type = [2,2])
    u_T_backward = Finite_Difference(initial_cond,[0,0],40,1000,(1.0,1.0,0.5),method = 'bd',b_type = [2,2])
    u_T_crank = Finite_Difference(initial_cond,[0,0],40,1000,(1.0,1.0,0.5),b_type = [2,2])
    if (u_T_forward and u_T_backward and u_T_crank) == [1,1]:
        print('Incorrect b_type test passed')
    else:
        print('Incorrect b_type test failed')

def test_incorrect_method():
    u_T = Finite_Difference(initial_cond,[0,0],10,1000,(1.0,1.0,0.5),method = 'fu',b_type = [0,0])
    if u_T == [1,1]:
        print('Incorrect method test passed')
    else:
        print('Incorrect method test failed')
if __name__ == '__main__':

    test_forward_output()
    test_backward_output()
    test_crank_output()
    test_forward_unstable()
    test_incorrect_b_type()
    test_incorrect_method()
