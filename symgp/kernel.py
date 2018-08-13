# coding: utf-8
from symfe import dx, Unknown, Constant
from symfe.core.basic import _coeffs_registery

from sympy import Function, Derivative, Symbol
from sympy import Tuple
from sympy import Expr, Basic, Add, Mul, Pow
from sympy import S
from sympy.core.function import UndefinedFunction
from sympy.core.function import AppliedUndef

# TODO a Kernel must be an extension of Function

class BasicKernel(Function):
    _name = None

    @property
    def name(self):
        return self._name

class UndefinedKernel(BasicKernel):

    def __new__(cls, name):

        obj = Basic.__new__(cls)
        obj._name = name
        return obj


class Kernel(BasicKernel):

    def __new__(cls, name, expr=None):

        if expr is None:
            return UndefinedKernel.__new__(cls, name)

        else:
            raise NotImplementedError('')

        obj = Basic.__new__(cls, expr)
        obj._name = name
        return obj

    @property
    def expr(self):
        return self._args[0]


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
    if not isinstance(K, Kernel):
        raise TypeError('Expecting a Kernel')

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

    # TODO like Function/undefinedFunction in sympy
#    if not isinstance(K, UndefinedKernel):
#        raise NotImplementedError('TODO')
#
#    else:
#        return F
