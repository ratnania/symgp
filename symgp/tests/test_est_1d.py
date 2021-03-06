# coding: utf-8
from sympy import Symbol
from sympy import Tuple
from sympy import lambdify
from sympy import Function

from symfe import dx, Unknown, Constant

from symgp.kernel import NLML

def test_est_1d_1():
    u = Unknown('u', ldim=1)
    phi = Constant('phi')

    # ... define a partial differential operator as a lambda function
    L = lambda u: dx(u) + phi*u
    L_expected = lambda u: dx(u) + 2.*u
    # ...

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

    us = u_num(x_u)
    fs = f_num(x_f)
    # ...

    # compute the likelihood
    nlml = NLML(L(u), u, 'RBF')

    # set values
    nlml.set_u(x_u, us)
    nlml.set_f(x_f, fs)

    from numpy.random import rand
    from numpy import exp, ones
    from time import time

    # ... using scipy
    from scipy.optimize import minimize

    x_start = rand(len(nlml.args))

    tb = time()
    m = minimize(nlml, x_start, method="Nelder-Mead")
    te = time()
    elapsed_scipy = te-tb

    args = exp(m.x)
    print('> estimated phi = ', nlml.map_args(args)['phi'])

    print('> elapsed time scipy  = ', elapsed_scipy)
    # ...

    # ... using pure python implementation
    from symgp.nelder_mead import nelder_mead

    x_start = rand(len(nlml.args))

    tb = time()
    m = nelder_mead(nlml, x_start,
                    step=0.1, no_improve_thr=10e-6, no_improv_break=10,
                    max_iter=0, alpha=1., gamma=2., rho=-0.5, sigma=0.5,
                    verbose=False)
    te = time()
    elapsed_python = te-tb

    args = exp(m[0])
    print('> estimated phi = ', nlml.map_args(args)['phi'])

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

    us = u_num(x_u)
    fs = f_num(x_f)
    # ...

    # compute the likelihood
    nlml = NLML(L(u), u, 'RBF')

    # set values
    nlml.set_u(x_u, us)
    nlml.set_f(x_f, fs)

    from numpy.random import rand
    from numpy import exp, ones
    from time import time

    # ... using pure python implementation
    from symgp.nelder_mead import nelder_mead

    x_start = rand(len(nlml.args))

    tb = time()
    m = nelder_mead(nlml, x_start,
                    step=0.1, no_improve_thr=10e-6, no_improv_break=10,
                    max_iter=0, alpha=1., gamma=2., rho=-0.5, sigma=0.5,
                    verbose=False)
    te = time()
    elapsed_python = te-tb

    args = exp(m[0])
    print('> estimated phi = ', nlml.map_args(args)['phi'])

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

    us = u_num(x_u)
    fs = f_num(x_f)
    # ...

    # compute the likelihood
    nlml = NLML(L(u), u, 'RBF')

    # set values
    nlml.set_u(x_u, us)
    nlml.set_f(x_f, fs)

    from numpy.random import rand
    from numpy import exp, ones
    from time import time

    # ... using pure python implementation
    from symgp.nelder_mead import nelder_mead

    x_start = rand(len(nlml.args))

    tb = time()
    m = nelder_mead(nlml, x_start,
                    step=0.1, no_improve_thr=10e-6, no_improv_break=10,
                    max_iter=0, alpha=1., gamma=2., rho=-0.5, sigma=0.5,
                    verbose=False)
    te = time()
    elapsed_python = te-tb

    args = exp(m[0])
    print('> estimated phi = ', nlml.map_args(args)['phi'])

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

    us = u_num(x_u)
    fs = f_num(x_f)
    # ...

    # compute the likelihood
    nlml = NLML(L(u), u, 'RBF')

    # set values
    nlml.set_u(x_u, us)
    nlml.set_f(x_f, fs)

    from numpy.random import rand
    from numpy import exp, ones
    from time import time

    # ... using pure python implementation
    from symgp.nelder_mead import nelder_mead

    x_start = rand(len(nlml.args))

    tb = time()
    m = nelder_mead(nlml, x_start,
                    step=0.1, no_improve_thr=10e-6, no_improv_break=10,
                    max_iter=0, alpha=1., gamma=2., rho=-0.5, sigma=0.5,
                    verbose=False)
    te = time()
    elapsed_python = te-tb

    args = exp(m[0])
    print('> estimated phi = ', nlml.map_args(args)['phi'])

    print('> elapsed time python = ', elapsed_python)
    # ...

def test_est_1d_5():
    u = Unknown('u', ldim=1)
    phi = Constant('phi')

    # ... define a partial differential operator as a lambda function
    L = lambda u: dx(u) + phi*u
    L_expected = lambda u: dx(u) + 2.*u
    # ...

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

    us = u_num(x_u)
    fs = f_num(x_f)
    # ...

    from numpy.random import rand
    from numpy import exp, ones
    from time import time

    def solve(kernel):
        print('>>>> using : ', kernel)

        # compute the likelihood
        nlml = NLML(L(u), u, kernel)

        # set values
        nlml.set_u(x_u, us)
        nlml.set_f(x_f, fs)

        x_start = rand(len(nlml.args))

        # ... using pure python implementation
        from symgp.nelder_mead import nelder_mead

        m = nelder_mead(nlml, x_start,
                        step=0.1, no_improve_thr=10e-6, no_improv_break=10,
                        max_iter=0, alpha=1., gamma=2., rho=-0.5, sigma=0.5,
                        verbose=False)

        args = exp(m[0])
        print('> estimated phi = ', nlml.map_args(args)['phi'])
#        print('> estimated args = ', nlml.map_args(args))
    # ...

    for kernel in ['RBF', 'SE', 'GammaSE', 'RQ', 'Linear', 'Periodic']:
        solve(kernel)

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

    from scipy.interpolate import interp1d
    unew = interp1d(xn, unew)
    un = interp1d(xn, un)

    fn = Function('fn')

    L = lambda u: u + Dt*fn(x)*dx(u) + nu*Dt*dx(dx(u))
    # ...

    # ... lambdification + evaluation
    from numpy import linspace, pi
    from numpy.random import rand

#    x_u = linspace(0, 2*pi, 30)

    x_u = rand(100) * 2*pi

    x_f = x_u

    us = un(x_u)
    fs = unew(x_f)
    # ...

    from numpy.random import rand
    from numpy import exp, ones, log
    from time import time
    from scipy.optimize import minimize
    from tabulate import tabulate

    # ...
    def solve(kernel):
        print('>>>> using : ', kernel)

        # compute the likelihood
        nlml = NLML(L(u), u, kernel)

        # set values
        nlml.set_u(x_u, us)
        nlml.set_f(x_f, fs)

        x_start = rand(len(nlml.args))

#        methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA']
#        methods = ['Nelder-Mead', 'Powell', 'CG']
        methods = ['Nelder-Mead']
        phis = []
        for method in methods:
            print('> {} method in progress ... '.format(method))
            m = minimize(nlml, x_start, method=method, jac=False)

            args = exp(m.x)
            phi_h = nlml.map_args(args)['nu']
            print('> estimated phi = ', phi_h)
            phis.append(phi_h)

        print(tabulate([phis], headers=methods))
    # ...

#    for kernel in ['RBF', 'SE', 'GammaSE', 'RQ', 'Linear', 'Periodic']:
#    for kernel in ['RBF', 'SE']:
    for kernel in ['SE']:
        solve(kernel)

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

    from scipy.interpolate import interp1d
    unew = interp1d(xn, unew)
    un = interp1d(xn, un)

    fn = Function('fn')

    L = lambda u: u + Dt*fn(x)*dx(u) + nu*Dt*dx(dx(u))
    # ...

    # ... lambdification + evaluation
    from numpy import linspace, pi
    from numpy.random import rand

#    x_u = linspace(0, 2*pi, 50)

    x_u = rand(50) * 2*pi

    x_f = x_u

    us = un(x_u)
    fs = unew(x_f)
    # ...

    from numpy.random import rand
    from numpy import exp, ones, log
    from time import time
    from scipy.optimize import minimize

    # compute the likelihood
    nlml = NLML(L(u), u, 'SE')

    # set values
    nlml.set_u(x_u, us)
    nlml.set_f(x_f, fs)

    x_start = rand(len(nlml.args))

    # ... using pure python implementation
    from symgp.nelder_mead import nelder_mead

    x_start = rand(len(nlml.args))

    m = nelder_mead(nlml, x_start,
                    step=1., no_improve_thr=1e-5, no_improv_break=6,
                    max_iter=0, alpha=1., gamma=1.5, rho=-0.5, sigma=.5,
                    verbose=False)

    args = exp(m[0])
    print('> estimated nu = ', nlml.map_args(args)['nu'])
    # ...

######################################
if __name__ == '__main__':
#    test_est_1d_1()
#    test_est_1d_2()
#    test_est_1d_3()
#    test_est_1d_4()
#    test_est_1d_5()

#    test_est_1d_6()
    test_est_1d_7()
