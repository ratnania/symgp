# coding: utf-8
from sympy import Symbol
from sympy import Tuple
from sympy import lambdify

from symfe import dx, Unknown, Constant

from symgp.kernel import RBF
from symgp.kernel import compile_nlml

def test_est_1d_1():
    u = Unknown('u', ldim=1)
    phi = Constant('phi')

    # ... define a partial differential operator as a lambda function
    L = lambda u: dx(u) + phi*u
    L_expected = lambda u: dx(u) + 2.*u
    # ...

    # compute the likelihood
    nlml = compile_nlml(L(u), u, RBF)

    # ... symbolic functions for unknown and rhs
    from sympy.abc import x
    from sympy import sin, cos

    u_sym = sin(x)
    f_sym = L_expected(u_sym)
    # ...

    # ... lambdification + evaluation
    from numpy import linspace, pi

    u_num = lambdify((x), u_sym, "numpy")
    f_num = lambdify((x), f_sym, "numpy")

    x_u = linspace(0, 2*pi, 10)
    x_f = x_u

    u = u_num(x_u)
    f = f_num(x_f)
    # ...

#    v = nlml((0.69, 1.), x_u, x_f, y_u, y_f, 1e-6)
#    print(v)


    nlml_wp = lambda params: nlml(params, x_u, x_f, u, f, 1e-6)

    from numpy.random import rand
    from numpy import exp, ones
    from time import time

    # ... using scipy
    from scipy.optimize import minimize

    tb = time()
    m = minimize(nlml_wp, rand(2), method="Nelder-Mead")
    te = time()
    elapsed_scipy = te-tb

    phi_h = exp(m.x)
    print(phi_h)

    print('> elapsed time scipy  = ', elapsed_scipy)
    # ...

    # ... using pure python implementation
    from symgp.nelder_mead import nelder_mead

    tb = time()
#    x_start = rand(2)
    x_start = ones(2)
    m = nelder_mead(nlml_wp, x_start,
                    step=0.1, no_improve_thr=10e-6, no_improv_break=10,
                    max_iter=0, alpha=1., gamma=2., rho=-0.5, sigma=0.5,
                    verbose=False)
    te = time()
    elapsed_python = te-tb

    phi_h = exp(m[0])
    print(phi_h)

    print('> elapsed time python = ', elapsed_python)
    # ...

def test_est_1d_2():
    u = Unknown('u', ldim=1)
    phi = Constant('phi')

    # ... define a partial differential operator as a lambda function
    from sympy.abc import x

    L = lambda u: x*dx(u) + phi*u
    L_expected = lambda u: x*dx(u) + 2.*u
    # ...

    # compute the likelihood
    nlml = compile_nlml(L(u), u, RBF)

    # ... symbolic functions for unknown and rhs
    from sympy.abc import x
    from sympy import sin, cos

    u_sym = sin(x)
    f_sym = L_expected(u_sym)
    # ...

    # ... lambdification + evaluation
    from numpy import linspace, pi

    u_num = lambdify((x), u_sym, "numpy")
    f_num = lambdify((x), f_sym, "numpy")

    x_u = linspace(0, 2*pi, 10)
    x_f = x_u

    u = u_num(x_u)
    f = f_num(x_f)
    # ...

#    v = nlml((0.69, 1.), x_u, x_f, y_u, y_f, 1e-6)
#    print(v)


    nlml_wp = lambda params: nlml(params, x_u, x_f, u, f, 1e-6)

    from numpy.random import rand
    from numpy import exp, ones
    from time import time

    # ... using pure python implementation
    from symgp.nelder_mead import nelder_mead

    tb = time()
#    x_start = rand(2)
    x_start = ones(2)
    m = nelder_mead(nlml_wp, x_start,
                    step=0.1, no_improve_thr=10e-6, no_improv_break=10,
                    max_iter=0, alpha=1., gamma=2., rho=-0.5, sigma=0.5,
                    verbose=False)
    te = time()
    elapsed_python = te-tb

    phi_h = exp(m[0])
    print(phi_h)

    print('> elapsed time python = ', elapsed_python)
    # ...

######################################
if __name__ == '__main__':
#    test_est_1d_1()
    test_est_1d_2()
