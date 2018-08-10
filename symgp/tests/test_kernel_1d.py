# coding: utf-8
from sympy import Function, Derivative, Symbol
from sympy import Tuple
from sympy import Expr, Basic, Add
from sympy.core.function import UndefinedFunction

from symfe import dx, Unknown, Constant

from symgp.kernel import Kernel
from symgp.kernel import evaluate


u = Unknown('u', ldim=1)
xi = Symbol('xi')
xj = Symbol('xj')
k = Function('K') # used for assert

def test_kernel_1d_1():
    L = dx(u)

    # ...
    expected = Derivative(k, xi)
    assert(evaluate(L, u, Kernel('K', xi)) == expected)
    # ...

    # ...
    expected = Derivative(k, xj)
    assert(evaluate(L, u, Kernel('K', xj)) == expected)
    # ...

    # ...
    expected = Derivative(k, xi, xj)
    assert(evaluate(L, u, Kernel('K', (xi, xj))) == expected)
    # ...

def test_kernel_1d_2():
    L = dx(dx(u))

    # ...
    expected = Derivative(k, xi, xi)
    assert(evaluate(L, u, Kernel('K', xi)) == expected)
    # ...

    # ...
    expected = Derivative(k, xj, xj)
    assert(evaluate(L, u, Kernel('K', xj)) == expected)
    # ...

    # ...
    expected = Derivative(k, xi, xi, xj, xj)
    assert(evaluate(L, u, Kernel('K', (xi, xj))) == expected)
    # ...

def test_kernel_1d_3():
    L = dx(u) + u

    # ...
    expected = k(xi) + Derivative(k, xi)
    assert(evaluate(L, u, Kernel('K', xi)) == expected)
    # ...

    # ...
    expected = k(xj) + Derivative(k, xj)
    assert(evaluate(L, u, Kernel('K', xj)) == expected)
    # ...

    # ...
    expected = (k(xi, xj) + Derivative(k, xi) + Derivative(k, xj) +
                Derivative(k, xi, xj))
    assert(evaluate(L, u, Kernel('K', (xi, xj))) == expected)
    # ...

def test_kernel_1d_4():
    L = dx(dx(u)) + dx(u) + u

    # ...
    expected = k(xi) + Derivative(k, xi) + Derivative(k, xi, xi)
    assert(evaluate(L, u, Kernel('K', xi)) == expected)
    # ...

    # ...
    expected = k(xj) + Derivative(k, xj) + Derivative(k, xj, xj)
    assert(evaluate(L, u, Kernel('K', xj)) == expected)
    # ...

    # ...
    expected = (k(xi, xj) + Derivative(k, xi) + Derivative(k, xj) +
                Derivative(k, xi, xi) + Derivative(k, xi, xj) +
                Derivative(k, xj, xj) + Derivative(k, xi, xi, xj) +
                Derivative(k, xi, xj, xj) + Derivative(k, xi, xi, xj, xj))
    assert(evaluate(L, u, Kernel('K', (xi, xj))) == expected)
    # ...

def test_kernel_1d_5():
    alpha = Constant('alpha')
    L = dx(u) + alpha*u

    # ...
    expected = alpha*k(xi) + Derivative(k, xi)
    assert(evaluate(L, u, Kernel('K', xi)) == expected)
    # ...

    # ...
    expected = alpha*k(xj) + Derivative(k, xj)
    assert(evaluate(L, u, Kernel('K', xj)) == expected)
    # ...

#    # ...
#    expected = (k(xi, xj) + Derivative(k, xi) + Derivative(k, xj) +
#                Derivative(k, xi, xj))
#    assert(evaluate(L, u, Kernel('K', (xi, xj))) == expected)
#    # ...

######################################
if __name__ == '__main__':

    test_kernel_1d_1()
    test_kernel_1d_2()
    test_kernel_1d_3()
    test_kernel_1d_4()
    test_kernel_1d_5()

#    c1 = Constant('c1')
#    c2 = Constant('c2')
#    c3 = Constant('c3')
#    L = c1*dx(dx(u)) + c2*dx(u) + c3*u
#    L = dx(dx(u)) + dx(u) + u

    alpha = Constant('alpha')
    L = dx(u) + alpha*u

#    Kij = evaluate(L, u, Kernel('K', (xi, xj)))
#    print(Kij)

#    Ki = evaluate(L, u, Kernel('K', xi))
#    print(Ki)
