# coding: utf-8
from sympy import Function, Derivative, Symbol
from sympy import Tuple
from sympy import Expr, Basic, Add
from sympy.core.function import UndefinedFunction
from sympy import exp
from sympy import simplify

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

    expected = theta_1*theta_2*theta_3*exp(-0.5*(xi - xj)**2)*exp(-0.5*(yi - yj)**2)*exp(-0.5*(zi - zj)**2)
    assert(K == expected)
    # ...

    # ...
    K = evaluate(L, u, Kernel('K'), (Tuple(xj, yj,zj)))
    K = update_kernel(K, RBF, ((xi,yi,zi), (xj,yj,zj)))

    expected = theta_1*theta_2*theta_3*exp(-0.5*(xi - xj)**2)*exp(-0.5*(yi - yj)**2)*exp(-0.5*(zi - zj)**2)
    assert(K == expected)
    # ...

    # ...
    K = evaluate(L, u, Kernel('K'), (Tuple(xi,yi,zi), Tuple(xj,yj,zj)))
    K = update_kernel(K, RBF, ((xi,yi,zi), (xj,yj,zj)))

    expected = theta_1*theta_2*theta_3*exp(-0.5*(xi - xj)**2)*exp(-0.5*(yi - yj)**2)*exp(-0.5*(zi - zj)**2)
    assert(K == expected)
    # ...

def test_kernel_3d_2():
    L = phi * u + dx(u) + dy(u) + dz(dz(u))

    # ...
    K = evaluate(L, u, Kernel('K'), (Tuple(xi, yi, zi)))
    K = update_kernel(K, RBF, ((xi,yi,zi), (xj,yj,zj)))

    expected = theta_1*theta_2*theta_3*(phi**3 + 1.0*phi**2*(-xi + xj) + 1.0*phi**2*(-yi + yj) + 1.0*phi**2*(-zi + zj) + 1.0*phi*(xi - xj)*(yi - yj) + 1.0*phi*(xi - xj)*(zi - zj) + 1.0*phi*(yi - yj)*(zi - zj) - 1.0*(xi - xj)*(yi - yj)*(zi - zj))*exp(-0.5*(xi - xj)**2)*exp(-0.5*(yi - yj)**2)*exp(-0.5*(zi - zj)**2)
    assert(simplify(K - expected) == 0)
    # ...

    # ...
    K = evaluate(L, u, Kernel('K'), (Tuple(xj, yj,zj)))
    K = update_kernel(K, RBF, ((xi,yi,zi), (xj,yj,zj)))

    expected = theta_1*theta_2*theta_3*(phi**3 + 1.0*phi**2*(xi - xj) + 1.0*phi**2*(yi - yj) + 1.0*phi**2*(zi - zj) + 1.0*phi*(xi - xj)*(yi - yj) + 1.0*phi*(xi - xj)*(zi - zj) + 1.0*phi*(yi - yj)*(zi - zj) + 1.0*(xi - xj)*(yi - yj)*(zi - zj))*exp(-0.5*(xi - xj)**2)*exp(-0.5*(yi - yj)**2)*exp(-0.5*(zi - zj)**2)
    assert(simplify(K - expected) == 0)
    # ...

    # ...
    K = evaluate(L, u, Kernel('K'), (Tuple(xi,yi,zi), Tuple(xj,yj,zj)))
    K = update_kernel(K, RBF, ((xi,yi,zi), (xj,yj,zj)))

    expected = theta_1*theta_2*theta_3*(phi**2 + 2.0*phi*((zi - zj)**2 - 1) - 1.0*(xi - xj)**2 - 2.0*(xi - xj)*(yi - yj) - 1.0*(yi - yj)**2 + 1.0*(zi - zj)**4 - 6.0*(zi - zj)**2 + 5.0)*exp(-0.5*(xi - xj)**2)*exp(-0.5*(yi - yj)**2)*exp(-0.5*(zi - zj)**2)
    assert(simplify(K - expected) == 0)
    # ...

######################################
if __name__ == '__main__':

    test_kernel_3d_1()
    test_kernel_3d_2()

##    L = u
#    L = phi * u + dx(u) + dy(u) + dz(dz(u))
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
