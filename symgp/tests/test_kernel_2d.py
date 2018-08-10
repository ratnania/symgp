# coding: utf-8
from sympy import Function, Derivative, Symbol
from sympy import Tuple
from sympy import Expr, Basic, Add
from sympy.core.function import UndefinedFunction

from symfe import dx, dy, Unknown, Constant

from symgp.kernel import Kernel
from symgp.kernel import evaluate


u = Unknown('u', ldim=2)
xi = Symbol('xi')
yi = Symbol('yi')
xj = Symbol('xj')
yj = Symbol('yj')
k = Function('K') # used for assert

def test_kernel_2d_1():
    L = dx(u) + dy(u)

#    # ...
#    expected = Derivative(k, xi)
#    assert(evaluate(L, u, Kernel('K', (Tuple(xi, yi))) == expected)
#    # ...
#
#    # ...
#    expected = Derivative(k, xj)
#    assert(evaluate(L, u, Kernel('K', (Tuple(xj, yj))) == expected)
#    # ...
#
#    # ...
#    expected = Derivative(k, xi, xj)
#    assert(evaluate(L, u, Kernel('K', (Tuple(xi,yi), Tuple(xj,yj)))) == expected)
#    # ...

######################################
if __name__ == '__main__':

#    test_kernel_2d_1()

    beta = Constant('beta')
    alpha = Constant('alpha')
    mu = Constant('mu')
    phi = Constant('phi')
#    L = beta*dx(dx(u)) + alpha*dx(u) + mu*u
#    L = dx(u) + dy(u)
#    L = u
#    L = dx(u)
#    L = dy(u)
#    L = dy(dy(u))
#    L = dx(dy(u))
    L = phi * u + dx(u) + dy(dy(u))

    Ki = evaluate(L, u, Kernel('K', (Tuple(xi, yi),)))
    print(Ki)

    Kij = evaluate(L, u, Kernel('K', (Tuple(xi,yi), Tuple(xj,yj))))
    print(Kij)
