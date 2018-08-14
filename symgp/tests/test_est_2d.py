# coding: utf-8
from sympy import Symbol
from sympy import Tuple
from sympy import lambdify

from symfe import dx, dy, Unknown, Constant

from symgp.kernel import RBF
from symgp.kernel import compile_nlml

######################################
if __name__ == '__main__':
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

    from scipy.optimize import minimize
    from numpy.random import rand
    from numpy import exp

    nlml_wp = lambda params: nlml(params, x_u, x_f, u, f, 1e-6)
    m = minimize(nlml_wp, rand(3), method="Nelder-Mead")
    phi_h = exp(m.x)
    print(phi_h)
