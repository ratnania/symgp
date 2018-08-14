# coding: utf-8
from sympy import Symbol
from sympy import Tuple
from sympy import lambdify

from symfe import dx, dy, Unknown, Constant

from symgp.kernel import RBF
from symgp.kernel import compile_nlml

def test_est_2d_1():
    u = Unknown('u', ldim=2)
    phi = Constant('phi')

    # ... define a partial differential operator as a lambda function
    L = lambda u: phi*u + dx(u) + dy(dy(u))
    L_expected = lambda u: 2.*u + dx(u) + dy(dy(u))
    # ...

    # compute the likelihood
    nlml = compile_nlml(L(u), u, RBF)

    # ... symbolic functions for unknown and rhs
    from sympy.abc import x, y
    from sympy import sin, cos

    u_sym = x**2 + y
    f_sym = L_expected(u_sym)
    # ...

    # ... lambdification + evaluation
    from numpy import linspace, meshgrid, zeros
    from numpy.random import rand

    u_num = lambdify((x,y), u_sym, "numpy")
    f_num = lambdify((x,y), f_sym, "numpy")

    t = linspace(0, 1, 5)
    x,y = meshgrid(t, t)
    x_u = zeros((x.size, 2))
    x_u[:,0] = x.reshape(x.size)
    x_u[:,1] = y.reshape(y.size)

#    x_u = rand(20,2)

    x_f = x_u

    u = u_num(x_u[:,0], x_u[:,1])
    f = f_num(x_f[:,0], x_f[:,1])
    # ...

#    v = nlml((0.69, 1, 1), x, x, y_u, y_f, 1e-6)
#    print(v)

    nlml_wp = lambda params: nlml(params, x_u, x_f, u, f, 1e-6)

    from numpy.random import rand
    from numpy import exp, ones, log
    from time import time

    # ... using pure python implementation
    from symgp.nelder_mead import nelder_mead

    tb = time()
    x_start = rand(3)
    x_start = ones(3)
    x_start[0] = rand()
#    x_start[0] = 0.6
    x_start[0] = 0.6
    print('> x_start = ', x_start)
    m = nelder_mead(nlml_wp, x_start,
                    step=0.1, no_improve_thr=10e-6, no_improv_break=10,
                    max_iter=0, alpha=1., gamma=2., rho=-0.5, sigma=0.5,
                    verbose=True)
    te = time()
    elapsed_python = te-tb

    phi_h = exp(m[0])
    print(phi_h)

    print('> elapsed time = ', elapsed_python)
    # ...

######################################
if __name__ == '__main__':
    test_est_2d_1()
