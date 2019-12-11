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
    u_T = Finite_Difference('forward',initial_cond,[0,0],40,1000,(1.0,1.0,0.5),b_type = [0,0],plot = True)
    if u_T == [1,1]:
        print('Forward unstable test passed')
    else:
        print('Forward unstable test failed')


if __name__ == '__main__':
    test_forward_unstable()
