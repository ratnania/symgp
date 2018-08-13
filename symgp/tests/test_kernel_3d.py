# coding: utf-8
from sympy import Function, Derivative, Symbol
from sympy import Tuple
from sympy import Expr, Basic, Add
from sympy.core.function import UndefinedFunction

from symfe import dx, dy, dz, Unknown, Constant

from symgp.kernel import Kernel
from symgp.kernel import evaluate


u = Unknown('u', ldim=3)
xi = Symbol('xi')
yi = Symbol('yi')
zi = Symbol('zi')
xj = Symbol('xj')
yj = Symbol('yj')
zj = Symbol('zj')
k = Function('K') # used for assert

def test_kernel_2d_1():
    L = u

    # ...
    expected = k(xi, yi, zi)
    assert(evaluate(L, u, Kernel('K'), (Tuple(xi,yi,zi),)) == expected)
    # ...

    # ...
    expected = k(xj, yj, zj)
    assert(evaluate(L, u, Kernel('K'), (Tuple(xj,yj,zj),)) == expected)
    # ...

    # ...
    expected = k(xi, yi, zi, xj, yj, zj)
    assert(evaluate(L, u, Kernel('K'), (Tuple(xi,yi,zi), Tuple(xj,yj,zj))) == expected)
    # ...

def test_kernel_2d_2():
    L = dx(u)

    # ...
    expected = Derivative(k, xi)
    assert(evaluate(L, u, Kernel('K'), (Tuple(xi,yi,zi),)) == expected)
    # ...

    # ...
    expected = Derivative(k, xj)
    assert(evaluate(L, u, Kernel('K'), (Tuple(xj,yj,zj),)) == expected)
    # ...

    # ...
    expected = Derivative(k, xi, xj)
    assert(evaluate(L, u, Kernel('K'), (Tuple(xi,yi,zi), Tuple(xj,yj,zj))) == expected)
    # ...

def test_kernel_2d_3():
    L = dy(u)

    # ...
    expected = Derivative(k, yi)
    assert(evaluate(L, u, Kernel('K'), (Tuple(xi,yi,zi),)) == expected)
    # ...

    # ...
    expected = Derivative(k, yj)
    assert(evaluate(L, u, Kernel('K'), (Tuple(xj,yj,zj),)) == expected)
    # ...

    # ...
    expected = Derivative(k, yi, yj)
    assert(evaluate(L, u, Kernel('K'), (Tuple(xi,yi,zi), Tuple(xj,yj,zj))) == expected)
    # ...

def test_kernel_2d_4():
    L = dz(u)

    # ...
    expected = Derivative(k, zi)
    assert(evaluate(L, u, Kernel('K'), (Tuple(xi,yi,zi),)) == expected)
    # ...

    # ...
    expected = Derivative(k, zj)
    assert(evaluate(L, u, Kernel('K'), (Tuple(xj,yj,zj),)) == expected)
    # ...

    # ...
    expected = Derivative(k, zi, zj)
    assert(evaluate(L, u, Kernel('K'), (Tuple(xi,yi,zi), Tuple(xj,yj,zj))) == expected)
    # ...

def test_kernel_2d_5():
    L = dx(u) + dy(u) + dz(u)

    # ...
    expected = Derivative(k, xi) + Derivative(k, yi) + Derivative(k, zi)
    assert(evaluate(L, u, Kernel('K'), (Tuple(xi,yi,zi),)) == expected)
    # ...

    # ...
    expected = Derivative(k, xj) + Derivative(k, yj) + Derivative(k, zj)
    assert(evaluate(L, u, Kernel('K'), (Tuple(xj,yj,zj),)) == expected)
    # ...

    # ...
    expected = (Derivative(k, xi, xj) + Derivative(k, xi, yj) +
                Derivative(k, xi, zj) + Derivative(k, yi, xj) +
                Derivative(k, yi, yj) + Derivative(k, yi, zj) +
                Derivative(k, zi, xj) + Derivative(k, zi, yj) +
                Derivative(k, zi, zj))
    assert(evaluate(L, u, Kernel('K'), (Tuple(xi,yi,zi), Tuple(xj,yj,zj))) == expected)
    # ...

def test_kernel_2d_6():
    L = dx(dx(u))

    # ...
    expected = Derivative(k, xi, xi)
    assert(evaluate(L, u, Kernel('K'), (Tuple(xi,yi,zi),)) == expected)
    # ...

    # ...
    expected = Derivative(k, xj, xj)
    assert(evaluate(L, u, Kernel('K'), (Tuple(xj,yj,zj),)) == expected)
    # ...

    # ...
    expected = Derivative(k, xi, xi, xj, xj)
    assert(evaluate(L, u, Kernel('K'), (Tuple(xi,yi,zi), Tuple(xj,yj,zj))) == expected)
    # ...

def test_kernel_2d_7():
    L = dy(dy(u))

    # ...
    expected = Derivative(k, yi, yi)
    assert(evaluate(L, u, Kernel('K'), (Tuple(xi,yi,zi),)) == expected)
    # ...

    # ...
    expected = Derivative(k, yj, yj)
    assert(evaluate(L, u, Kernel('K'), (Tuple(xj,yj,zj),)) == expected)
    # ...

    # ...
    expected = Derivative(k, yi, yi, yj, yj)
    assert(evaluate(L, u, Kernel('K'), (Tuple(xi,yi,zi), Tuple(xj,yj,zj))) == expected)
    # ...

def test_kernel_2d_8():
    L = dz(dz(u))

    # ...
    expected = Derivative(k, zi, zi)
    assert(evaluate(L, u, Kernel('K'), (Tuple(xi,yi,zi),)) == expected)
    # ...

    # ...
    expected = Derivative(k, zj, zj)
    assert(evaluate(L, u, Kernel('K'), (Tuple(xj,yj,zj),)) == expected)
    # ...

    # ...
    expected = Derivative(k, zi, zi, zj, zj)
    assert(evaluate(L, u, Kernel('K'), (Tuple(xi,yi,zi), Tuple(xj,yj,zj))) == expected)
    # ...

def test_kernel_2d_9():
    L = dx(dy(u))

    # ...
    expected = Derivative(k, yi, xi)
    assert(evaluate(L, u, Kernel('K'), (Tuple(xi,yi,zi),)) == expected)
    # ...

    # ...
    expected = Derivative(k, yj, xj)
    assert(evaluate(L, u, Kernel('K'), (Tuple(xj,yj,zj),)) == expected)
    # ...

    # ...
    expected = Derivative(k, yi, xi, yj, xj)
    assert(evaluate(L, u, Kernel('K'), (Tuple(xi,yi,zi), Tuple(xj,yj,zj))) == expected)
    # ...

def test_kernel_2d_10():
    L = dx(dx(u)) + dy(dy(u)) + dz(dz(u))

    # ...
    expected = Derivative(k, xi, xi) + Derivative(k, yi, yi) + Derivative(k, zi, zi)
    assert(evaluate(L, u, Kernel('K'), (Tuple(xi,yi,zi),)) == expected)
    # ...

    # ...
    expected = Derivative(k, xj, xj) + Derivative(k, yj, yj) + Derivative(k, zj, zj)
    assert(evaluate(L, u, Kernel('K'), (Tuple(xj,yj,zj),)) == expected)
    # ...

    # ...
    expected = (Derivative(k, xi, xi, xj, xj) + Derivative(k, xi, xi, yj, yj) +
                Derivative(k, xi, xi, zj, zj) + Derivative(k, yi, yi, xj, xj) +
                Derivative(k, yi, yi, yj, yj) + Derivative(k, yi, yi, zj, zj) +
                Derivative(k, zi, zi, xj, xj) + Derivative(k, zi, zi, yj, yj) +
                Derivative(k, zi, zi, zj, zj))
    assert(evaluate(L, u, Kernel('K'), (Tuple(xi,yi,zi), Tuple(xj,yj,zj))) == expected)
    # ...

def test_kernel_2d_11():
    phi = Constant('phi')
    L = phi * u + dx(u) + dy(u) + dz(dz(u))

    # ...
    expected = phi*k(xi, yi, zi) + Derivative(k, xi) + Derivative(k, yi) + Derivative(k, zi, zi)
    assert(evaluate(L, u, Kernel('K'), (Tuple(xi,yi,zi),)) == expected)
    # ...

    # ...
    expected = phi*k(xj, yj, zj) + Derivative(k, xj) + Derivative(k, yj) + Derivative(k, zj, zj)
    assert(evaluate(L, u, Kernel('K'), (Tuple(xj,yj,zj),)) == expected)
    # ...

    # ...
    expected = (phi**2*k(xi, yi, zi, xj, yj, zj) + phi*Derivative(k, xi) + phi*Derivative(k, xj) +
                phi*Derivative(k, yi) + phi*Derivative(k, yj) + phi*Derivative(k, zi, zi) +
                phi*Derivative(k, zj, zj) + Derivative(k, xi, xj) + Derivative(k, xi, yj) +
                Derivative(k, yi, xj) + Derivative(k, yi, yj) + Derivative(k, xi, zj, zj) +
                Derivative(k, yi, zj, zj) + Derivative(k, zi, zi, xj) +
                Derivative(k, zi, zi, yj) + Derivative(k, zi, zi, zj, zj))
    assert(evaluate(L, u, Kernel('K'), (Tuple(xi,yi,zi), Tuple(xj,yj,zj))) == expected)
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
    test_kernel_2d_10()
    test_kernel_2d_11()

#    beta = Constant('beta')
#    alpha = Constant('alpha')
#    mu = Constant('mu')
#    phi = Constant('phi')
#
#    L = phi * u + dx(u) + dy(u) + dz(dz(u))
#
#    Ki = evaluate(L, u, Kernel('K', (Tuple(xi, yi, zi),)))
#    print(Ki)
#
#    Kj = evaluate(L, u, Kernel('K', (Tuple(xj, yj, zj),)))
#    print(Kj)
#
#    Kij = evaluate(L, u, Kernel('K', (Tuple(xi,yi,zi), Tuple(xj,yj,zj))))
#    print(Kij)
