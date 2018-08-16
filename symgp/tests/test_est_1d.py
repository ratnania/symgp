# coding: utf-8
from sympy import Symbol
from sympy import Tuple
from sympy import lambdify
from sympy import Function

from symfe import dx, Unknown, Constant

from symgp.kernel import RBF, GRBF, RQuad, ExpSin, DotProduct
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

def test_est_1d_3():
    u = Unknown('u', ldim=1)
    phi = Constant('phi')

    # ... define a partial differential operator as a lambda function
    from sympy.abc import x
    from sympy import sin, cos

    L = lambda u: sin(x)*dx(u) + phi*cos(x)*u
    L_expected = lambda u: sin(x)*dx(u) + 2.*cos(x)*u
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

def test_est_1d_4():
    u = Unknown('u', ldim=1)
    phi = Constant('phi')

    # ... define a partial differential operator as a lambda function
    from sympy.abc import x
    from sympy import sin, cos

    L = lambda u: dx(u) + cos(phi*x)*u
    L_expected = lambda u: dx(u) + cos(2.*x)*u
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
    x_start = rand(2)
    x_start[0] = 0.9
#    print(x_start)
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

def test_est_1d_5():
    u = Unknown('u', ldim=1)
    phi = Constant('phi')

    # ... define a partial differential operator as a lambda function
    L = lambda u: dx(u) + phi*u
    L_expected = lambda u: dx(u) + 2.*u
    # ...

    # compute the likelihood
    nlml = compile_nlml(L(u), u, GRBF)
#    nlml = compile_nlml(L(u), u, RQuad)
#    nlml = compile_nlml(L(u), u, ExpSin)
#    nlml = compile_nlml(L(u), u, DotProduct)

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

    x_start = rand(3)
#    x_start = rand(2)

    # ... using scipy
    from scipy.optimize import minimize

    tb = time()
    m = minimize(nlml_wp, x_start, method="Nelder-Mead")
    te = time()
    elapsed_scipy = te-tb

    phi_h = exp(m.x)
    print(phi_h)

    print('> elapsed time scipy  = ', elapsed_scipy)
    # ...

    # ... using pure python implementation
    from symgp.nelder_mead import nelder_mead

    tb = time()
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

def test_est_1d_6():
    """Explicit time step for Burgers"""

    u = Unknown('u', ldim=1)
    nu = Constant('nu')

    # ... define a partial differential operator as a lambda function
    from sympy.abc import x
    from sympy import sin, cos

    Dt = 0.0010995574287564279
    nu_expected = 0.07

    from numpy import genfromtxt
    xn = genfromtxt('x.txt')
    unew = genfromtxt('unew.txt')
    un = genfromtxt('un.txt')
#    un = genfromtxt('fn.txt')

    from scipy.interpolate import interp1d
    unew = interp1d(xn, unew)
    un = interp1d(xn, un)

    fn = Function('fn')

    L = lambda u: u + Dt*fn(x)*dx(u) + nu*Dt*dx(dx(u))
    # ...

    # compute the likelihood
    nlml = compile_nlml(L(u), u, RBF)

    # ... lambdification + evaluation
    from numpy import linspace, pi

    x_u = linspace(0, 2*pi, 30)
    x_f = x_u

    u = un(x_u)
    f = unew(x_f)

#    from numpy.linalg import norm
#    norm_u = norm(u)
#    norm_f = norm(f)
#    u = u/norm_u
#    f = f/norm_f
    # ...

    nlml_wp = lambda params: nlml(params, x_u, x_f, u, f, 1e-6)

    from numpy.random import rand
    from numpy import exp, ones, log
    from time import time

    # ... using pure python implementation
    from symgp.nelder_mead import nelder_mead

    x_start = rand(2)
#    x_start[0] = log(0.1)
#    print('> x_start = ', x_start)

#    m = nelder_mead(nlml_wp, x_start,
#                    step=.2, no_improve_thr=10e-3, no_improv_break=10,
#                    max_iter=0,
#                    alpha=[1., 1.],
#                    gamma=[4., 2.],
#                    rho=[-1., -0.5],
#                    sigma=[1., 0.5],
#                    verbose=False)
#
#    params = exp(m[0])
#    phi_h = params[0]
#    theta_h = params[1]
#    print('> estimated nu = ', phi_h)
#    print('> estimated theta = ', theta_h)
    # ...

    # ...
    from scipy.optimize import minimize

    x_start = rand(2)
    x_start[0] = log(0.1)
    print('> x_start = ', x_start)

    methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B',
               'TNC', 'COBYLA']
    phis = []
    for method in methods:
        print('> {} method in progress ... '.format(method))
        m = minimize(nlml_wp, x_start, method=method)
        params = exp(m.x)

        phi_h = params[0]
        phis.append(phi_h)
    # ...

    from tabulate import tabulate
    print(tabulate([phis], headers=methods))


def test_est_1d_7():
    """Explicit time step for Burgers"""

    u = Unknown('u', ldim=1)
    nu = Constant('nu')

    # ... define a partial differential operator as a lambda function
    from sympy.abc import x
    from sympy import sin, cos

    Dt = 0.0010995574287564279
    nu_expected = 0.07

    from numpy import genfromtxt
    xn = genfromtxt('x.txt')
    unew = genfromtxt('unew.txt')
    un = genfromtxt('un.txt')
#    un = genfromtxt('fn.txt')

    from scipy.interpolate import interp1d
    unew = interp1d(xn, unew)
    un = interp1d(xn, un)

    fn = Function('fn')

    L = lambda u: u + Dt*fn(x)*dx(u) + nu*Dt*dx(dx(u))
    # ...

    # compute the likelihood
    nlml = compile_nlml(L(u), u, GRBF)

    # ... lambdification + evaluation
    from numpy import linspace, pi

    x_u = linspace(0, 2*pi, 30)
    x_f = x_u

    u = un(x_u)
    f = unew(x_f)

#    from numpy.linalg import norm
#    norm_u = norm(u)
#    norm_f = norm(f)
#    u = u/norm_u
#    f = f/norm_f
    # ...

    nlml_wp = lambda params: nlml(params, x_u, x_f, u, f, 1e-6)

    from numpy.random import rand
    from numpy import exp, ones, log
    from time import time

    # ... using pure python implementation
    from symgp.nelder_mead import nelder_mead

#    x_start = rand(3)
#    x_start[0] = log(0.1)
#    print('> x_start = ', x_start)

#    m = nelder_mead(nlml_wp, x_start,
#                    step=.2, no_improve_thr=10e-3, no_improv_break=10,
#                    max_iter=0,
#                    alpha=[1., 1.],
#                    gamma=[4., 2.],
#                    rho=[-1., -0.5],
#                    sigma=[1., 0.5],
#                    verbose=False)
#
#    params = exp(m[0])
#    phi_h = params[0]
#    theta_h = params[1]
#    print('> estimated nu = ', phi_h)
#    print('> estimated theta = ', theta_h)
    # ...

    # ...
    from scipy.optimize import minimize

    x_start = rand(3)
#    x_start[0] = log(0.1)
    print('> x_start = ', x_start)

    methods = ['Nelder-Mead', 'Powell', 'TNC']
    phis = []
    for method in methods:
        print('> {} method in progress ... '.format(method))
        m = minimize(nlml_wp, x_start, method=method)
        params = exp(m.x)

        phi_h = params[1]
        phis.append(phi_h)
    # ...

    from tabulate import tabulate
    print(tabulate([phis], headers=methods))



######################################
if __name__ == '__main__':
#    test_est_1d_1()
#    test_est_1d_2()
#    test_est_1d_3()
#    test_est_1d_4()
    test_est_1d_5()

#    test_est_1d_6()
#    test_est_1d_7()
