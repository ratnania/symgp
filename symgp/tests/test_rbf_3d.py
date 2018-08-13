# coding: utf-8
from sympy import Function, Derivative, Symbol
from sympy import Tuple
from sympy import Expr, Basic, Add
from sympy.core.function import UndefinedFunction
from sympy import exp

from symfe import dx, dy, dz, Unknown, Constant

from symgp.kernel import Kernel
from symgp.kernel import RBF
from symgp.kernel import evaluate
from symgp.kernel import update_kernel


u = Unknown('u', ldim=3)
xi = Symbol('xi')
yi = Symbol('yi')
zi = Symbol('zi')
xj = Symbol('xj')
yj = Symbol('yj')
zj = Symbol('zj')
theta_1 = Constant('theta_1')
theta_2 = Constant('theta_2')
theta_3 = Constant('theta_3')
phi = Constant('phi')

def test_kernel_3d_1():
    L = u

    # ...
    K = evaluate(L, u, Kernel('K'), (Tuple(xi, yi, zi)))
    K = update_kernel(K, RBF, ((xi,yi,zi), (xj,yj,zj)))

    expected = exp(-theta_1*(xi - xj)**2 - theta_2*(yi - yj)**2 - theta_3*(zi - zj)**2)
    assert(K == expected)
    # ...

    # ...
    K = evaluate(L, u, Kernel('K'), (Tuple(xj, yj,zj)))
    K = update_kernel(K, RBF, ((xi,yi,zi), (xj,yj,zj)))

    expected = exp(-theta_1*(xi - xj)**2 - theta_2*(yi - yj)**2 - theta_3*(zi - zj)**2)
    assert(K == expected)
    # ...

    # ...
    K = evaluate(L, u, Kernel('K'), (Tuple(xi,yi,zi), Tuple(xj,yj,zj)))
    K = update_kernel(K, RBF, ((xi,yi,zi), (xj,yj,zj)))

    expected = exp(-theta_1*(xi - xj)**2 - theta_2*(yi - yj)**2 - theta_3*(zi - zj)**2)
    assert(K == expected)
    # ...

def test_kernel_3d_2():
    L = phi * u + dx(u) + dy(u) + dz(dz(u))

    # ...
    K = evaluate(L, u, Kernel('K'), (Tuple(xi, yi, zi)))
    K = update_kernel(K, RBF, ((xi,yi,zi), (xj,yj,zj)))

    expected = (phi**3 - phi**2*theta_1*(2*xi - 2*xj) - phi**2*theta_2*(2*yi - 2*yj) - phi**2*theta_3*(2*zi - 2*zj) + 4*phi*theta_1*theta_2*(xi - xj)*(yi - yj) + 4*phi*theta_1*theta_3*(xi - xj)*(zi - zj) + 4*phi*theta_2*theta_3*(yi - yj)*(zi - zj) - 8*theta_1*theta_2*theta_3*(xi - xj)*(yi - yj)*(zi - zj))*exp(-theta_1*(xi - xj)**2 - theta_2*(yi - yj)**2 - theta_3*(zi - zj)**2)
    assert(K == expected)
    # ...

    # ...
    K = evaluate(L, u, Kernel('K'), (Tuple(xj, yj,zj)))
    K = update_kernel(K, RBF, ((xi,yi,zi), (xj,yj,zj)))

    expected = (phi**3 - phi**2*theta_1*(-2*xi + 2*xj) - phi**2*theta_2*(-2*yi + 2*yj) - phi**2*theta_3*(-2*zi + 2*zj) + 4*phi*theta_1*theta_2*(xi - xj)*(yi - yj) + 4*phi*theta_1*theta_3*(xi - xj)*(zi - zj) + 4*phi*theta_2*theta_3*(yi - yj)*(zi - zj) + 8*theta_1*theta_2*theta_3*(xi - xj)*(yi - yj)*(zi - zj))*exp(-theta_1*(xi - xj)**2 - theta_2*(yi - yj)**2 - theta_3*(zi - zj)**2)
    assert(K == expected)
    # ...

    # ...
    K = evaluate(L, u, Kernel('K'), (Tuple(xi,yi,zi), Tuple(xj,yj,zj)))
    K = update_kernel(K, RBF, ((xi,yi,zi), (xj,yj,zj)))

    expected = (phi**2 - phi*theta_1*(-2*xi + 2*xj) - phi*theta_1*(2*xi - 2*xj) - phi*theta_2*(-2*yi + 2*yj) -
                phi*theta_2*(2*yi - 2*yj) + 4*phi*theta_3*(2*theta_3*(zi - zj)**2 - 1) -
                8*theta_1*theta_2*(xi - xj)*(yi - yj) +
                4*theta_1*theta_3*(xi - xj)*(-2*theta_3*(zi - zj)**2 + 1) +
                4*theta_1*theta_3*(xi - xj)*(2*theta_3*(zi - zj)**2 - 1) +
                2*theta_1*(-2*theta_1*(xi - xj)**2 + 1) +
                4*theta_2*theta_3*(yi - yj)*(-2*theta_3*(zi - zj)**2 + 1) +
                4*theta_2*theta_3*(yi - yj)*(2*theta_3*(zi - zj)**2 - 1) +
                2*theta_2*(-2*theta_2*(yi - yj)**2 + 1) +
                4*theta_3**2*(4*theta_3**2*(zi - zj)**4 -
                              12*theta_3*(zi - zj)**2 + 3))*exp(-theta_1*(xi - xj)**2 - theta_2*(yi - yj)**2 - theta_3*(zi - zj)**2)
    assert(K == expected)
    # ...

######################################
if __name__ == '__main__':

    test_kernel_3d_1()
    test_kernel_3d_2()

#    L = phi * u + dx(u) + dy(u) + dz(dz(u))
##    L = u
#
#    K = evaluate(L, u, Kernel('K'), (Tuple(xi, yi, zi)))
#    K = update_kernel(K, RBF, ((xi,yi,zi), (xj,yj,zj)))
#    print(K)
#
#    K = evaluate(L, u, Kernel('K'), (Tuple(xj, yj, zj)))
#    K = update_kernel(K, RBF, ((xi,yi,zi), (xj,yj,zj)))
#    print(K)
#
#    K = evaluate(L, u, Kernel('K'), (Tuple(xi,yi,zi), Tuple(xj,yj,zj)))
#    K = update_kernel(K, RBF, ((xi,yi,zi), (xj,yj,zj)))
#    print(K)
