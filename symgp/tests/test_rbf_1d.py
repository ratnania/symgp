# coding: utf-8
from sympy import Function, Derivative, Symbol
from sympy import Tuple
from sympy import Expr, Basic, Add
from sympy.core.function import UndefinedFunction
from sympy import exp

from symfe import dx, Unknown, Constant

from symgp.kernel import Kernel
from symgp.kernel import RBF
from symgp.kernel import evaluate
from symgp.kernel import update_kernel


u = Unknown('u', ldim=1)
xi = Symbol('xi')
xj = Symbol('xj')
theta = Constant('theta')
alpha = Constant('alpha')

def test_kernel_1d_1():
    L = u

    # ...
    K = evaluate(L, u, Kernel('K'), xi)
    K = update_kernel(K, RBF, (xi, xj))

    expected = theta*exp(-0.5*(xi - xj)**2)
    assert(K == expected)
    # ...

    # ...
    K = evaluate(L, u, Kernel('K'), xj)
    K = update_kernel(K, RBF, (xi, xj))

    expected = theta*exp(-0.5*(xi - xj)**2)
    assert(K == expected)
    # ...

    # ...
    K = evaluate(L, u, Kernel('K'), (xi, xj))
    K = update_kernel(K, RBF, (xi, xj))

    expected = theta*exp(-0.5*(xi - xj)**2)
    assert(K == expected)
    # ...

def test_kernel_1d_2():
    L = dx(u) + alpha*u

    # ...
    K = evaluate(L, u, Kernel('K'), xi)
    K = update_kernel(K, RBF, (xi, xj))

    expected = theta*(alpha - 1.0*xi + 1.0*xj)*exp(-0.5*(xi - xj)**2)
    assert(K == expected)
    # ...

    # ...
    K = evaluate(L, u, Kernel('K'), xj)
    K = update_kernel(K, RBF, (xi, xj))

    expected = theta*(alpha + 1.0*xi - 1.0*xj)*exp(-0.5*(xi - xj)**2)
    assert(K == expected)
    # ...

    # ...
    K = evaluate(L, u, Kernel('K'), (xi, xj))
    K = update_kernel(K, RBF, (xi, xj))

    expected = theta*(alpha**2 + alpha*(-1.0*xi + 1.0*xj) + alpha*(1.0*xi - 1.0*xj) - 1.0*(xi - xj)**2 + 1.0)*exp(-0.5*(xi - xj)**2)
    assert(K == expected)
    # ...

######################################
if __name__ == '__main__':

    test_kernel_1d_1()
    test_kernel_1d_2()

#    L =  u
#    L = dx(u) + alpha*u
#
#    K = evaluate(L, u, Kernel('K'), xi)
#    K = update_kernel(K, RBF, (xi, xj))
#    print(K)
#
#    K = evaluate(L, u, Kernel('K'), xj)
#    K = update_kernel(K, RBF, (xi, xj))
#    print(K)
#
#    K = evaluate(L, u, Kernel('K'), (xi, xj))
#    K = update_kernel(K, RBF, (xi, xj))
#    print(K)
