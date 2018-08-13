# coding: utf-8
from sympy import Function, Derivative, Symbol
from sympy import Tuple
from sympy import Expr, Basic, Add
from sympy.core.function import UndefinedFunction
from sympy import exp

from symfe import dx, dy, Unknown, Constant

from symgp.kernel import RBF
from symgp.kernel import compile_kernels
from symgp.kernel import compile_nlml


u = Unknown('u', ldim=2)
xi = Symbol('xi')
yi = Symbol('yi')
xj = Symbol('xj')
yj = Symbol('yj')
theta = Constant('theta')
phi = Constant('phi')




######################################
if __name__ == '__main__':

    L = phi * u + dx(u) + dy(dy(u))

#    d = compile_kernels(L, u, RBF, (Tuple(xi,yi), Tuple(xj,yj)))

    nlml = compile_nlml(L, u, RBF, (Tuple(xi,yi), Tuple(xj,yj)))

#    K = evaluate(L, u, Kernel('K'), xi)
#    K = update_kernel(K, RBF, (xi, xj))
#    print(K)

    import numpy as np
    import sympy as sp
    from scipy.optimize import minimize
    import matplotlib.pyplot as plt

    x = np.random.rand(20,2)
    y_u = np.multiply(x[:,0], x[:,1]) - x[:,1]**2
    y_f = 2.0*y_u + x[:,1] - 2

#    v = nlml((0.69, 1, 1), x, x, y_u, y_f, 1e-6)
#    print(v)

    nlml_wp = lambda params: nlml(params, x, x, y_u, y_f, 1e-7)
    m = minimize(nlml_wp, np.random.rand(3), method="Nelder-Mead")
    phi_h = np.exp(m.x)
    print(phi_h)
