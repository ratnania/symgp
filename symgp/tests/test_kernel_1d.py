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

    # ...
    expected = (alpha**2*k(xi, xj) + alpha*Derivative(k, xi) +
                alpha*Derivative(k, xj) + Derivative(k, xi, xj))
    assert(evaluate(L, u, Kernel('K', (xi, xj))) == expected)
    # ...

def test_kernel_1d_6():
    beta = Constant('beta')
    alpha = Constant('alpha')
    mu = Constant('mu')
    L = beta*dx(dx(u)) + alpha*dx(u) + mu*u

    # ...
    expected = alpha*Derivative(k, xi) + beta*Derivative(k, xi, xi) + mu*k(xi)
    assert(evaluate(L, u, Kernel('K', xi)) == expected)
    # ...

    # ...
    expected = alpha*Derivative(k, xj) + beta*Derivative(k, xj, xj) + mu*k(xj)
    assert(evaluate(L, u, Kernel('K', xj)) == expected)
    # ...

    # ...
    expected = (alpha**2*Derivative(k, xi, xj) +
                alpha*beta*Derivative(k, xi, xi, xj) +
                alpha*beta*Derivative(k, xi, xj, xj) +
                alpha*mu*Derivative(k, xi) +
                alpha*mu*Derivative(k, xj) +
                beta**2*Derivative(k, xi, xi, xj, xj) +
                beta*mu*Derivative(k, xi, xi) +
                beta*mu*Derivative(k, xj, xj) +
                mu**2*k(xi, xj))
    assert(evaluate(L, u, Kernel('K', (xi, xj))) == expected)
    # ...

######################################
if __name__ == '__main__':

    test_kernel_1d_1()
    test_kernel_1d_2()
    test_kernel_1d_3()
    test_kernel_1d_4()
    test_kernel_1d_5()
    test_kernel_1d_6()

#    beta = Constant('beta')
#    alpha = Constant('alpha')
#    mu = Constant('mu')
#    L = beta*dx(dx(u)) + alpha*dx(u) + mu*u

#    Ki = evaluate(L, u, Kernel('K', xi))
#    print(Ki)
#
#    Kj = evaluate(L, u, Kernel('K', xj))
#    print(Kj)
#
#    Kij = evaluate(L, u, Kernel('K', (xi, xj)))
#    print(Kij)
