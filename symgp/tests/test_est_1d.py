# coding: utf-8
from sympy import Symbol
from sympy import Tuple
from sympy import lambdify

from symfe import dx, Unknown, Constant

from symgp.kernel import RBF
from symgp.kernel import compile_nlml

def test_est_1d_1():
    u = Unknown('u', ldim=1)
    xi = Symbol('xi')
    xj = Symbol('xj')
    phi = Constant('phi')

    # ... define a partial differential operator as a lambda function
    L = lambda u: dx(u) + phi*u
    L_expected = lambda u: dx(u) + 2.*u
    # ...

    # compute the likelihood
    nlml = compile_nlml(L(u), u, RBF, (xi, xj))

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
    x_f = linspace(0, 2*pi, 10)

    u = u_num(x_u)
    f = f_num(x_f)
    # ...

#    v = nlml((0.69, 1.), x_u, x_f, y_u, y_f, 1e-6)
#    print(v)

    from scipy.optimize import minimize
    from numpy.random import rand
    from numpy import exp

    nlml_wp = lambda params: nlml(params, x_u, x_f, u, f, 1e-6)
    m = minimize(nlml_wp, rand(2), method="Nelder-Mead")
    phi_h = exp(m.x)
    print(phi_h)


######################################
if __name__ == '__main__':
    test_est_1d_1()
