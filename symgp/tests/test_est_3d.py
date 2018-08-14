# coding: utf-8
from sympy import Symbol
from sympy import Tuple
from sympy import lambdify

from symfe import dx, dy, dz, Unknown, Constant

from symgp.kernel import RBF
from symgp.kernel import compile_nlml

def test_est_3d_1():
    u = Unknown('u', ldim=3)
    phi = Constant('phi')

    # ... define a partial differential operator as a lambda function
    L = lambda u: phi*u + dx(u) + dy(u) + dz(dz(u))
    L_expected = lambda u: 2.*u + dx(u) + dy(u) + dz(dz(u))
    # ...

    # compute the likelihood
    nlml = compile_nlml(L(u), u, RBF)

    # ... symbolic functions for unknown and rhs
    from sympy.abc import x, y, z
    from sympy import sin, cos

    u_sym = x**2 + y*z
    f_sym = L_expected(u_sym)
    # ...

    # ... lambdification + evaluation
    from numpy import linspace, meshgrid, zeros
    from numpy.random import rand

    u_num = lambdify((x,y,z), u_sym, "numpy")
    f_num = lambdify((x,y,z), f_sym, "numpy")

    t = linspace(0, 1, 3)
    x,y,z = meshgrid(t, t, t)
    x_u = zeros((x.size, 3))
    x_u[:,0] = x.reshape(x.size)
    x_u[:,1] = y.reshape(y.size)
    x_u[:,2] = z.reshape(z.size)

#    x_u = rand(20,2)

    x_f = x_u

    u = u_num(x_u[:,0], x_u[:,1], x_u[:,2])
    f = f_num(x_f[:,0], x_f[:,1], x_f[:,2])
    # ...

#    v = nlml((0.69, 1, 1), x, x, y_u, y_f, 1e-6)
#    print(v)

    nlml_wp = lambda params: nlml(params, x_u, x_f, u, f, 1e-6)

    from numpy.random import rand
    from numpy import exp, ones
    from time import time

    # ... using pure python implementation
    from symgp.nelder_mead import nelder_mead

    tb = time()
    x_start = rand(4)
#    x_start = ones(4)
    print('> x_start = ', x_start)
    m = nelder_mead(nlml_wp, x_start,
                    step=0.1*x_start.max(), no_improve_thr=10e-4,
                    no_improv_break=4,
                    max_iter=0, alpha=.5, gamma=1.5, rho=-0.5, sigma=0.5,
                    verbose=False)
    te = time()
    elapsed_python = te-tb

    phi_h = exp(m[0])
    print(phi_h)

    print('> elapsed time = ', elapsed_python)
    # ...

######################################
if __name__ == '__main__':
    test_est_3d_1()
