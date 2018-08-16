# coding: utf-8
from sympy import Function, Derivative, Symbol
from sympy import Tuple
from sympy import Expr, Basic, Add
from sympy.core.function import UndefinedFunction
from sympy import exp
from sympy import simplify

from symfe import dx, dy, Unknown, Constant

from symgp.kernel import Kernel
from symgp.kernel import RBF
from symgp.kernel import evaluate
from symgp.kernel import update_kernel

u = Unknown('u', ldim=2)
xi = Symbol('xi')
yi = Symbol('yi')
xj = Symbol('xj')
yj = Symbol('yj')
theta_1 = Constant('theta_1')
theta_2 = Constant('theta_2')
phi = Constant('phi')

def test_kernel_2d_1():
    L = u

    # ...
    K = evaluate(L, u, Kernel('K'), (Tuple(xi, yi)))
    K = update_kernel(K, RBF, ((xi,yi), (xj,yj)))

    expected = theta_1*theta_2*exp(-0.5*(xi - xj)**2)*exp(-0.5*(yi - yj)**2)
    assert(K == expected)
    # ...

    # ...
    K = evaluate(L, u, Kernel('K'), (Tuple(xj, yj)))
    K = update_kernel(K, RBF, ((xi,yi), (xj,yj)))

    expected = theta_1*theta_2*exp(-0.5*(xi - xj)**2)*exp(-0.5*(yi - yj)**2)
    assert(K == expected)
    # ...

    # ...
    K = evaluate(L, u, Kernel('K'), (Tuple(xi,yi), Tuple(xj,yj)))
    K = update_kernel(K, RBF, ((xi,yi), (xj,yj)))

    expected = theta_1*theta_2*exp(-0.5*(xi - xj)**2)*exp(-0.5*(yi - yj)**2)
    assert(K == expected)
    # ...

def test_kernel_2d_2():
    L = phi * u + dx(u) + dy(dy(u))

    # ...
    K = evaluate(L, u, Kernel('K'), (Tuple(xi, yi)))
    K = update_kernel(K, RBF, ((xi,yi), (xj,yj)))

    expected = theta_1*theta_2*(phi**2 - 1.0*phi*(xi - xj) - 1.0*phi*(yi - yj) + 1.0*(xi - xj)*(yi - yj))*exp(-0.5*(xi - xj)**2)*exp(-0.5*(yi - yj)**2)
    assert(simplify(K - expected) == 0)
    # ...

    # ...
    K = evaluate(L, u, Kernel('K'), (Tuple(xj, yj)))
    K = update_kernel(K, RBF, ((xi,yi), (xj,yj)))

    expected = theta_1*theta_2*(phi**2 + 1.0*phi*(xi - xj) + 1.0*phi*(yi - yj) + 1.0*(xi - xj)*(yi - yj))*exp(-0.5*(xi - xj)**2)*exp(-0.5*(yi - yj)**2)
    assert(simplify(K - expected) == 0)
    # ...

    # ...
    K = evaluate(L, u, Kernel('K'), (Tuple(xi,yi), Tuple(xj,yj)))
    K = update_kernel(K, RBF, ((xi,yi), (xj,yj)))

    expected = theta_1*theta_2*(phi**2 + 2.0*phi*((yi - yj)**2 - 1) - 1.0*(xi - xj)**2 + 1.0*(yi - yj)**4 - 6.0*(yi - yj)**2 + 4.0)*exp(-0.5*(xi - xj)**2)*exp(-0.5*(yi - yj)**2)
    assert(simplify(K - expected) == 0)
    # ...

######################################
if __name__ == '__main__':

    test_kernel_2d_1()
    test_kernel_2d_2()

#    L = u
#    L = phi * u + dx(u) + dy(dy(u))
#
#    K = evaluate(L, u, Kernel('K'), (Tuple(xi, yi)))
#    K = update_kernel(K, RBF, ((xi,yi), (xj,yj)))
#    print(K)
#
#    K = evaluate(L, u, Kernel('K'), (Tuple(xj, yj)))
#    K = update_kernel(K, RBF, ((xi,yi), (xj,yj)))
#    print(K)
#
#    K = evaluate(L, u, Kernel('K'), (Tuple(xi,yi), Tuple(xj,yj)))
#    K = update_kernel(K, RBF, ((xi,yi), (xj,yj)))
#    print(K)
