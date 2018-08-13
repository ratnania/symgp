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
    L = u

    # ...
    expected = k(xi, yi)
    assert(evaluate(L, u, Kernel('K', (Tuple(xi, yi),))) == expected)
    # ...

    # ...
    expected = k(xj, yj)
    assert(evaluate(L, u, Kernel('K', (Tuple(xj, yj),))) == expected)
    # ...

    # ...
    expected = k(xi, yi, xj, yj)
    assert(evaluate(L, u, Kernel('K', (Tuple(xi,yi), Tuple(xj,yj)))) == expected)
    # ...

def test_kernel_2d_2():
    L = dx(u)

    # ...
    expected = Derivative(k, xi)
    assert(evaluate(L, u, Kernel('K', (Tuple(xi, yi),))) == expected)
    # ...

    # ...
    expected = Derivative(k, xj)
    assert(evaluate(L, u, Kernel('K', (Tuple(xj, yj),))) == expected)
    # ...

    # ...
    expected = Derivative(k, xi, xj)
    assert(evaluate(L, u, Kernel('K', (Tuple(xi,yi), Tuple(xj,yj)))) == expected)
    # ...

def test_kernel_2d_3():
    L = dy(u)

    # ...
    expected = Derivative(k, yi)
    assert(evaluate(L, u, Kernel('K', (Tuple(xi, yi),))) == expected)
    # ...

    # ...
    expected = Derivative(k, yj)
    assert(evaluate(L, u, Kernel('K', (Tuple(xj, yj),))) == expected)
    # ...

    # ...
    expected = Derivative(k, yi, yj)
    assert(evaluate(L, u, Kernel('K', (Tuple(xi,yi), Tuple(xj,yj)))) == expected)
    # ...

def test_kernel_2d_4():
    L = dx(u) + dy(u)

    # ...
    expected = Derivative(k, xi) + Derivative(k, yi)
    assert(evaluate(L, u, Kernel('K', (Tuple(xi, yi),))) == expected)
    # ...

    # ...
    expected = Derivative(k, xj) + Derivative(k, yj)
    assert(evaluate(L, u, Kernel('K', (Tuple(xj, yj),))) == expected)
    # ...

    # ...
    expected = (Derivative(k, xi, xj) +
                Derivative(k, xi, yj) +
                Derivative(k, yi, xj) +
                Derivative(k, yi, yj))
    assert(evaluate(L, u, Kernel('K', (Tuple(xi,yi), Tuple(xj,yj)))) == expected)
    # ...

def test_kernel_2d_5():
    L = dx(dx(u))

    # ...
    expected = Derivative(k, xi, xi)
    assert(evaluate(L, u, Kernel('K', (Tuple(xi, yi),))) == expected)
    # ...

    # ...
    expected = Derivative(k, xj, xj)
    assert(evaluate(L, u, Kernel('K', (Tuple(xj, yj),))) == expected)
    # ...

    # ...
    expected = Derivative(k, xi, xi, xj, xj)
    assert(evaluate(L, u, Kernel('K', (Tuple(xi,yi), Tuple(xj,yj)))) == expected)
    # ...

def test_kernel_2d_6():
    L = dy(dy(u))

    # ...
    expected = Derivative(k, yi, yi)
    assert(evaluate(L, u, Kernel('K', (Tuple(xi, yi),))) == expected)
    # ...

    # ...
    expected = Derivative(k, yj, yj)
    assert(evaluate(L, u, Kernel('K', (Tuple(xj, yj),))) == expected)
    # ...

    # ...
    expected = Derivative(k, yi, yi, yj, yj)
    assert(evaluate(L, u, Kernel('K', (Tuple(xi,yi), Tuple(xj,yj)))) == expected)
    # ...

def test_kernel_2d_7():
    L = dx(dy(u))

    # ...
    expected = Derivative(k, yi, xi)
    assert(evaluate(L, u, Kernel('K', (Tuple(xi, yi),))) == expected)
    # ...

    # ...
    expected = Derivative(k, yj, xj)
    assert(evaluate(L, u, Kernel('K', (Tuple(xj, yj),))) == expected)
    # ...

    # ...
    expected = Derivative(k, yi, xi, yj, xj)
    assert(evaluate(L, u, Kernel('K', (Tuple(xi,yi), Tuple(xj,yj)))) == expected)
    # ...

def test_kernel_2d_8():
    phi = Constant('phi')
    L = phi * u + dx(u) + dy(dy(u))

    # ...
    expected = phi*k(xi, yi) + Derivative(k, xi) + Derivative(k, yi, yi)
    assert(evaluate(L, u, Kernel('K', (Tuple(xi, yi),))) == expected)
    # ...

    # ...
    expected = phi*k(xj, yj) + Derivative(k, xj) + Derivative(k, yj, yj)
    assert(evaluate(L, u, Kernel('K', (Tuple(xj, yj),))) == expected)
    # ...

    # ...
    expected = (phi*Derivative(k, xi) + phi*Derivative(k, xj) +
                phi*Derivative(k, yi, yi) + phi*Derivative(k, yj, yj) + Derivative(k, xi, xj) +
                Derivative(k, xi, yj, yj) + Derivative(k, yi, yi, xj) +
                phi**2*k(xi, yi, xj, yj) +Derivative(k, yi, yi, yj, yj))
    assert(evaluate(L, u, Kernel('K', (Tuple(xi,yi), Tuple(xj,yj)))) == expected)
    # ...

def test_kernel_2d_9():
    beta = Constant('beta')
    alpha = Constant('alpha')
    mu = Constant('mu')
    L = beta*dx(dx(u)) + alpha*dx(u) + mu*u

    # ...
    expected = alpha*Derivative(k, xi) + beta*Derivative(k, xi, xi) + mu*k(xi, yi)
    assert(evaluate(L, u, Kernel('K', (Tuple(xi, yi),))) == expected)
    # ...

    # ...
    expected = alpha*Derivative(k, xj) + beta*Derivative(k, xj, xj) + mu*k(xj, yj)
    assert(evaluate(L, u, Kernel('K', (Tuple(xj, yj),))) == expected)
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
                mu**2*k(xi, yi, xj, yj))
    assert(evaluate(L, u, Kernel('K', (Tuple(xi,yi), Tuple(xj,yj)))) == expected)
    # ...

######################################
if __name__ == '__main__':

    test_kernel_2d_1()
    test_kernel_2d_2()
    test_kernel_2d_3()
    test_kernel_2d_4()
    test_kernel_2d_5()
    test_kernel_2d_6()
    test_kernel_2d_7()
    test_kernel_2d_8()
    test_kernel_2d_9()

#    beta = Constant('beta')
#    alpha = Constant('alpha')
#    mu = Constant('mu')
#    phi = Constant('phi')
#    L = beta*dx(dx(u)) + alpha*dx(u) + mu*u
#
#    Ki = evaluate(L, u, Kernel('K', (Tuple(xi, yi),)))
#    print(Ki)
#
#    Kj = evaluate(L, u, Kernel('K', (Tuple(xj, yj),)))
#    print(Kj)
#
#    Kij = evaluate(L, u, Kernel('K', (Tuple(xi,yi), Tuple(xj,yj))))
#    print(Kij)
