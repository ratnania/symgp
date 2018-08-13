# coding: utf-8
from sympy import Function, Derivative, Symbol
from sympy import Tuple
from sympy import Expr, Basic, Add
from sympy.core.function import UndefinedFunction
from sympy import exp

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

def test_kernel_1d_1():
    L = u

    # ...
    K = evaluate(L, u, Kernel('K'), (Tuple(xi, yi)))
    K = update_kernel(K, RBF((xi,yi), (xj,yj)))

    expected = exp(-theta_1*(xi - xj)**2 - theta_2*(yi - yj)**2)
    assert(K == expected)
    # ...

    # ...
    K = evaluate(L, u, Kernel('K'), (Tuple(xj, yj)))
    K = update_kernel(K, RBF((xi,yi), (xj,yj)))

    expected = exp(-theta_1*(xi - xj)**2 - theta_2*(yi - yj)**2)
    assert(K == expected)
    # ...

    # ...
    K = evaluate(L, u, Kernel('K'), (Tuple(xi,yi), Tuple(xj,yj)))
    K = update_kernel(K, RBF((xi,yi), (xj,yj)))

    expected = exp(-theta_1*(xi - xj)**2 - theta_2*(yi - yj)**2)
    assert(K == expected)
    # ...

def test_kernel_1d_2():
    L = phi * u + dx(u) + dy(dy(u))

    # ...
    K = evaluate(L, u, Kernel('K'), (Tuple(xi, yi)))
    K = update_kernel(K, RBF((xi,yi), (xj,yj)))

    expected = (phi**2*exp(-theta_1*(xi - xj)**2 - theta_2*(yi - yj)**2) -
                phi*theta_1*(2*xi - 2*xj)*exp(-theta_1*(xi - xj)**2 - theta_2*(yi - yj)**2) -
                phi*theta_2*(2*yi - 2*yj)*exp(-theta_1*(xi - xj)**2 - theta_2*(yi - yj)**2) +
                4*theta_1*theta_2*(xi - xj)*(yi - yj)*exp(-theta_1*(xi - xj)**2 - theta_2*(yi - yj)**2))
    assert(K == expected)
    # ...

    # ...
    K = evaluate(L, u, Kernel('K'), (Tuple(xj, yj)))
    K = update_kernel(K, RBF((xi,yi), (xj,yj)))

    expected = (phi**2*exp(-theta_1*(xi - xj)**2 - theta_2*(yi - yj)**2) -
                phi*theta_1*(-2*xi + 2*xj)*exp(-theta_1*(xi - xj)**2 - theta_2*(yi - yj)**2) -
                phi*theta_2*(-2*yi + 2*yj)*exp(-theta_1*(xi - xj)**2 - theta_2*(yi - yj)**2) +
                4*theta_1*theta_2*(xi - xj)*(yi - yj)*exp(-theta_1*(xi - xj)**2 - theta_2*(yi - yj)**2))
    assert(K == expected)
    # ...

    # ...
    K = evaluate(L, u, Kernel('K'), (Tuple(xi,yi), Tuple(xj,yj)))
    K = update_kernel(K, RBF((xi,yi), (xj,yj)))

    expected = (phi**2*exp(-theta_1*(xi - xj)**2 - theta_2*(yi - yj)**2) -
                phi*theta_1*(-2*xi + 2*xj)*exp(-theta_1*(xi - xj)**2 - theta_2*(yi - yj)**2) -
                phi*theta_1*(2*xi - 2*xj)*exp(-theta_1*(xi - xj)**2 - theta_2*(yi - yj)**2) +
                4*phi*theta_2*(2*theta_2*(yi - yj)**2 - 1)*exp(-theta_1*(xi - xj)**2 - theta_2*(yi - yj)**2)
                + 4*theta_1*theta_2*(xi - xj)*(-2*theta_2*(yi - yj)**2 + 1)*exp(-theta_1*(xi - xj)**2 - theta_2*(yi - yj)**2) +
                4*theta_1*theta_2*(xi - xj)*(2*theta_2*(yi - yj)**2 - 1)*exp(-theta_1*(xi - xj)**2 - theta_2*(yi - yj)**2) +
                2*theta_1*(-2*theta_1*(xi - xj)**2 + 1)*exp(-theta_1*(xi - xj)**2 - theta_2*(yi - yj)**2) +
                4*theta_2**2*(4*theta_2**2*(yi - yj)**4 - 12*theta_2*(yi - yj)**2 +
                              3)*exp(-theta_1*(xi - xj)**2 - theta_2*(yi - yj)**2))
    assert(K == expected)
    # ...

######################################
if __name__ == '__main__':

    test_kernel_1d_1()
    test_kernel_1d_2()

#    L = phi * u + dx(u) + dy(dy(u))
#
#    K = evaluate(L, u, Kernel('K'), (Tuple(xi, yi)))
#    K = update_kernel(K, RBF((xi,yi), (xj,yj)))
#    print(K)
#
#    K = evaluate(L, u, Kernel('K'), (Tuple(xj, yj)))
#    K = update_kernel(K, RBF((xi,yi), (xj,yj)))
#    print(K)
#
#    K = evaluate(L, u, Kernel('K'), (Tuple(xi,yi), Tuple(xj,yj)))
#    K = update_kernel(K, RBF((xi,yi), (xj,yj)))
#    print(K)
