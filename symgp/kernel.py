# coding: utf-8
from symfe import dx, Unknown, Constant
from symfe.core.basic import _coeffs_registery

from sympy import Function, Derivative, Symbol
from sympy import Tuple
from sympy import Expr, Basic, Add, Mul, Pow
from sympy import S
from sympy import exp
from sympy.core.function import UndefinedFunction
from sympy.core.function import AppliedUndef
from sympy import collect


class KernelBase(Function):
    _name = 'K'

    def __new__(cls, *args, **options):
        obj = Function.__new__(cls, *args, **options)
        return obj

    @property
    def name(self):
        return self._name


class RBF(KernelBase):
    _name = 'RBF'

    @classmethod
    def eval(cls, *args):

        if not( len(args) == 2 ):
            raise ValueError('> Expecting two arguments')

        if isinstance(args[0], Symbol):
            ldim = 1

        elif isinstance(args[0], (tuple, list, Tuple)):
            ldim = len(args[0])

        # TODO must check that all arguments are of the same type (Symbols or
        # tuples)

        if ldim == 1:
            theta = Constant('theta')
            xi,xj = args
            expr = theta*exp(-1/(2)*((xi - xj)**2))

        elif ldim == 2:
            theta_1 = Constant('theta_1')
            theta_2 = Constant('theta_2')
            xi,yi = args[0]
            xj,yj = args[1]
            expr = exp(- theta_1 * (xi - xj)**2 - theta_2 * (yi - yj)**2)

        elif ldim == 3:
            theta_1 = Constant('theta_1')
            theta_2 = Constant('theta_2')
            theta_3 = Constant('theta_3')
            xi,yi,zi = args[0]
            xj,yj,zj = args[1]
            expr = exp(- theta_1 * (xi - xj)**2 - theta_2 * (yi - yj)**2 - theta_3 * (zi - zj)**2)

        return expr


Kernel = KernelBase


def _evaluate(expr, u, K, xi, x):

    if isinstance(expr, Add):
        args = [_evaluate(a, u, K, xi, x) for a in expr.args]
        return Add(*args)

    # TODO remove try/except
    try:
        L = expr.subs({u: K})
    except:
        L = expr

#    print('> ', L, type(L))
    if isinstance(L, Derivative):
        f = L.args[0] ; args = list(L.variables)
        if isinstance(f, AppliedUndef):
            f = f.func

        args = Tuple(*args)
        for _x, _xi in zip(x, xi):
            args = args.subs(_x, _xi)

        return Derivative(f, *args)

    elif isinstance(L, UndefinedFunction):
        return L(*xi)

    elif isinstance(L, AppliedUndef):
        args = list(L.args)
        args += xi
        func = L.func
        return func(*args)

    elif isinstance(L, Mul):
        coeffs  = [i for i in L.args if isinstance(i, _coeffs_registery)]
        vectors = [i for i in L.args if not(i in coeffs)]

        i = S.One
        if coeffs:
            i = Mul(*coeffs)

        j = S.One
        if vectors:
            args = [_evaluate(a, u, K, xi, x) for a in vectors]
            j = Mul(*args)

        return Mul(i, j)

    elif isinstance(expr, Pow):

        if isinstance(expr.base, _coeffs_registery):
            return expr
        else:
            raise NotImplementedError('')

#            b = _evaluate(a, u, K, xi, x)
#            b = atomize(expr.base, dim=dim)
#            e = expr.exp
#
#            return Pow(b, e)

    else:
        print(L)
        raise NotImplementedError('{}'.format(type(L)))

def evaluate(expr, u, K, variables):

    coordinates = ['x', 'y', 'z']
    coordinates = [Symbol(i) for i in coordinates]

    # ...
    if isinstance(variables, str):
        variables = [Symbol(variables)]

    elif isinstance(variables, Symbol):
        variables = [variables]

    if isinstance(variables, (tuple, list, Tuple)):
        ls = []
        for v in variables:
            if isinstance(v, str):
                v = Symbol(v)
                v = [v]

            elif isinstance(v, Symbol):
                v = [v]

            elif isinstance(v, (tuple, list, Tuple)):
                for a in v:
                    if not isinstance(a, (str, Symbol)):
                        print(type(a))
                        raise TypeError('expecing str or Symbol')

                vs = []
                for i in v:
                    if isinstance(i, str):
                        vs.append(Symbol(i))
                    elif isinstance(i, Symbol):
                        vs.append(i)
                v = vs

            v = Tuple(*v)
            ls.append(v)

        variables = Tuple(*ls)
    # ...

    # ... TODO improve this. we should pass F = K, and it must work
#    F = K
    F = Function(K.name)
    # ...

    for xis in variables:
        xi = xis ; x = coordinates[:len(xis)]

        if isinstance(F, Add):
            args = [_evaluate(expr, u, f, xi, x) for f in F.args]
            F = Add(*args)

        elif isinstance(F, Mul):
            coeffs  = [i for i in F.args if isinstance(i, _coeffs_registery)]
            vectors = [i for i in F.args if not(i in coeffs)]

            i = S.One
            if coeffs:
                i = Mul(*coeffs)

            j = S.One
            if vectors:
                args = [_evaluate(a, u, K, xi, x) for a in vectors]
                j = Mul(*args)

            F = Mul(i, j)

        else:
            F = _evaluate(expr, u, F, xi, x)

    return F

def _update_kernel(expr, kernel):

    if isinstance(expr, Add):
        args = [_update_kernel(a, kernel) for a in expr.args]
        return Add(*args)

    elif isinstance(expr, Mul):
        coeffs  = [i for i in expr.args if isinstance(i, _coeffs_registery)]
        vectors = [i for i in expr.args if not(i in coeffs)]

        i = S.One
        if coeffs:
            i = Mul(*coeffs)

        j = S.One
        if vectors:
            args = [_update_kernel(a, kernel) for a in vectors]
            j = Mul(*args)

        return Mul(i, j)

    elif isinstance(expr, Derivative):
        f = expr.args[0] ; args = list(expr.variables)

        args = Tuple(*args)
        expr = Derivative(kernel, *args)
        return expr.doit()
    # ...

    elif isinstance(expr, AppliedUndef):
        # TODO use arguments
        return kernel

    return expr

def update_kernel(expr, kernel, variables):
    expr = _update_kernel(expr, kernel(*variables))

    cls_name = kernel(evaluate=False)
    if isinstance(cls_name, RBF):
        expr = expr.subs(RBF(*variables), Symbol('RBF'))
        expr = collect(expr, Symbol('RBF'))
        expr = expr.subs(Symbol('RBF'), RBF(*variables))

    return expr
