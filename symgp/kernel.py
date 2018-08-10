# coding: utf-8
from symfe import dx, Unknown, Constant

from sympy import Function, Derivative, Symbol
from sympy import Tuple
from sympy import Expr, Basic, Add
from sympy.core.function import UndefinedFunction

class Kernel(Expr):

    def __new__(cls, name, variables, expr=None):
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
                            raise TypeError('expecing str or Symbol')

                    v = [Symbol(i) for i in v]

                v = Tuple(*v)
                ls.append(v)

            variables = Tuple(*ls)
        # ...

        return Basic.__new__(cls, name, variables, expr)

    @property
    def name(self):
        return self._args[0]

    @property
    def variables(self):
        return self._args[1]

    @property
    def expr(self):
        return self._args[2]


def evaluate_core(expr, u, K, xi, x):
    if isinstance(expr, Add):
        args = [evaluate_core(a, u, K, xi, x) for a in expr.args]
        return Add(*args)

    L = expr.subs({u: K})
    if isinstance(L, Derivative):
        f = L.args[0] ; args = list(L.variables)
        args = Tuple(*args)
        args = args.subs(x, xi)
        return Derivative(f, *args)

    elif isinstance(L, UndefinedFunction):
        return L(xi)

    else:
        raise NotImplementedError('{}'.format(type(L)))

def evaluate(expr, u, K):
    if not isinstance(K, Kernel):
        raise TypeError('Expecting a Kernel')

    coordinates = ['x', 'y', 'z']
    coordinates = [Symbol(i) for i in coordinates]

    variables = K.variables
    F = Function(K.name)
    for xis in variables:
        for xi, x in zip(xis, coordinates):
            F = evaluate_core(expr, u, F, xi, x)

    if K.expr:
        raise NotImplemented('TODO')

    else:
        return F
