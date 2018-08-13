# coding: utf-8
from sympy import Function, Derivative, Symbol
from sympy import Tuple
from sympy import Expr, Basic, Add
from sympy.core.function import UndefinedFunction
from sympy import exp

from symfe import dx, Unknown, Constant

from symgp.kernel import RBF
from symgp.kernel import compile_kernels
from symgp.kernel import compile_nlml


u = Unknown('u', ldim=1)
xi = Symbol('xi')
xj = Symbol('xj')
theta = Constant('theta')
phi = Constant('phi')




######################################
if __name__ == '__main__':

    L = dx(u) + phi*u

#    d = compile_kernels(L, u, RBF, (xi, xj))
    nlml = compile_nlml(L, u, RBF, (xi, xj))

#    K = evaluate(L, u, Kernel('K'), xi)
#    K = update_kernel(K, RBF, (xi, xj))
#    print(K)

    import numpy as np
    import sympy as sp
    from scipy.optimize import minimize
    import matplotlib.pyplot as plt

    x_u = np.linspace(0,2*np.pi,10)
    y_u = np.sin(x_u)
    x_f = np.linspace(0,2*np.pi, 10)
    y_f = 2.0*np.sin(x_f) + np.cos(x_f)

    v = nlml((0.69, 1.), x_u, x_f, y_u, y_f, 1e-6)
    print(v)

    nlml_wp = lambda params: nlml(params, x_u, x_f, y_u, y_f, 1e-6)
    m = minimize(nlml_wp, np.random.rand(2), method="Nelder-Mead")
    phi_h = np.exp(m.x)
    print(phi_h)
