# coding: utf-8
from sympy import Symbol
from sympy import Tuple
from sympy import lambdify

from symfe import dx, dy, Unknown, Constant

from symgp.kernel import NLML

def test_est_2d_1():
    u = Unknown('u', ldim=2)
    phi = Constant('phi')

    # ... define a partial differential operator as a lambda function
    L = lambda u: phi*u + dx(u) + dy(dy(u))
    L_expected = lambda u: 2.*u + dx(u) + dy(dy(u))
    # ...

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

    x_f = x_u

    us = u_num(x_u[:,0], x_u[:,1])
    fs = f_num(x_f[:,0], x_f[:,1])
    # ...

    # compute the likelihood
    nlml = NLML(L(u), u, 'SE')

    # set values
    nlml.set_u(x_u, us)
    nlml.set_f(x_f, fs)

    from numpy.random import rand
    from numpy import exp, ones, log

    # ... using pure python implementation
    from symgp.nelder_mead import nelder_mead

    x_start = rand(len(nlml.args))
    print('> x_start = ', x_start)
    m = nelder_mead(nlml, x_start,
                    step=0.1, no_improve_thr=10e-4,
                    no_improv_break=4,
                    max_iter=0, alpha=.5, gamma=1.5, rho=-0.5, sigma=0.5,
                    verbose=False)

    args = exp(m[0])
    print('> estimated phi = ', nlml.map_args(args)['phi'])
    # ...

######################################
if __name__ == '__main__':
    test_est_2d_1()
