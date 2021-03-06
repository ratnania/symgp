# coding: utf-8
from symfe import dx, Unknown, Constant
from symfe.core.basic import _coeffs_registery
from symfe.codegen.utils import known_math_functions, known_math_constants
from symfe.codegen.utils import print_import_from_numpy

from sympy import Function, Derivative, Symbol
from sympy import NumberSymbol
from sympy import Tuple
from sympy import Expr, Basic, Add, Mul, Pow
from sympy import S
from sympy import exp, sin, pi, Abs
from sympy.core.function import UndefinedFunction
from sympy.core.function import AppliedUndef
from sympy import collect
from sympy import lambdify
from sympy import simplify
from sympy import preorder_traversal

from numpy import asarray, unique

import matplotlib.pyplot as plt

iteration = 0

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
            theta_1 = Constant('theta_1')
            xi,xj = args
            expr = theta_1 * exp(-0.5*(xi - xj)**2)

        elif ldim == 2:
            theta_1 = Constant('theta_1')
            theta_2 = Constant('theta_2')
            xi,yi = args[0]
            xj,yj = args[1]
            expr_1 = theta_1 * exp(-0.5*(xi - xj)**2)
            expr_2 = theta_2 * exp(-0.5*(yi - yj)**2)
            expr = expr_1 * expr_2

        elif ldim == 3:
            theta_1 = Constant('theta_1')
            theta_2 = Constant('theta_2')
            theta_3 = Constant('theta_3')
            xi,yi,zi = args[0]
            xj,yj,zj = args[1]
            expr_1 = theta_1 * exp(-0.5*(xi - xj)**2)
            expr_2 = theta_2 * exp(-0.5*(yi - yj)**2)
            expr_3 = theta_3 * exp(-0.5*(zi - zj)**2)
            expr = expr_1 * expr_2 * expr_3

        return expr

class SE(KernelBase):
    _name = 'SE'

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
            theta_1 = Constant('theta_1')
            l_1 = Constant('l_1')
            xi,xj = args
            expr = theta_1*exp(-1/(2)*((xi - xj)**2/l_1**2))

        elif ldim == 2:
            theta_1 = Constant('theta_1')
            theta_2 = Constant('theta_2')
            l_1 = Constant('l_1')
            l_2 = Constant('l_2')
            xi,yi = args[0]
            xj,yj = args[1]
            expr_1 = theta_1 * exp(-0.5*(xi - xj)**2/l_1**2)
            expr_2 = theta_2 * exp(-0.5*(yi - yj)**2/l_2**2)
            expr = expr_1 * expr_2

        elif ldim == 3:
            theta_1 = Constant('theta_1')
            theta_2 = Constant('theta_2')
            theta_3 = Constant('theta_3')
            l_1 = Constant('l_1')
            l_2 = Constant('l_2')
            l_3 = Constant('l_3')
            xi,yi,zi = args[0]
            xj,yj,zj = args[1]
            expr_1 = theta_1 * exp(-0.5*(xi - xj)**2/l_1**2)
            expr_2 = theta_2 * exp(-0.5*(yi - yj)**2/l_2**2)
            expr_3 = theta_3 * exp(-0.5*(zi - zj)**2/l_3**2)
            expr = expr_1 * expr_2 * expr_3

        return expr

class CSE(KernelBase):
    _name = 'CSE'

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
            raise NotImplementedError('')

        elif ldim == 2:
            theta_1 = Constant('theta_1')
            theta_2 = Constant('theta_2')
            l_1 = Constant('l_1')
            l_2 = Constant('l_2')
            l_12 = Constant('l_12')
            xi,yi = args[0]
            xj,yj = args[1]
            expr_1 = theta_1 * exp(-0.5*(xi - xj)**2/l_1**2)
            expr_2 = theta_2 * exp(-0.5*(yi - yj)**2/l_2**2)
            expr_12 = exp(-(xi - xj)*(yi - yj)/l_12)
            expr = expr_1 * expr_2 * expr_12

        elif ldim == 3:
            raise NotImplementedError('')

        return expr

class GammaSE(KernelBase):
    _name = 'GammaSE'

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
            theta_1 = Constant('theta_1')
            gamma_1 = Constant('gamma_1')
            l_1 = Constant('l_1')
            xi,xj = args
            expr = theta_1*exp(-1/(2)*(((xi - xj)**2)**(gamma_1/2)/l_1**2))

        elif ldim == 2:
            theta_1 = Constant('theta_1')
            theta_2 = Constant('theta_2')
            gamma_1 = Constant('gamma_1')
            gamma_2 = Constant('gamma_2')
            l_1 = Constant('l_1')
            l_2 = Constant('l_2')
            xi,yi = args[0]
            xj,yj = args[1]
            expr_1 = theta_1 * exp(-0.5*((xi - xj)**2)**(gamma_1/2)/l_1**2)
            expr_2 = theta_2 * exp(-0.5*((yi - yj)**2)**(gamma_2/2)/l_2**2)
            expr = expr_1 * expr_2

        elif ldim == 3:
            theta_1 = Constant('theta_1')
            theta_2 = Constant('theta_2')
            theta_3 = Constant('theta_3')
            gamma_1 = Constant('gamma_1')
            gamma_2 = Constant('gamma_2')
            gamma_3 = Constant('gamma_3')
            l_1 = Constant('l_1')
            l_2 = Constant('l_2')
            l_3 = Constant('l_3')
            xi,yi,zi = args[0]
            xj,yj,zj = args[1]
            expr_1 = theta_1 * exp(-0.5*((xi - xj)**2)**(gamma_1/2)/l_1**2)
            expr_2 = theta_2 * exp(-0.5*((yi - yj)**2)**(gamma_2/2)/l_2**2)
            expr_3 = theta_3 * exp(-0.5*((zi - zj)**2)**(gamma_3/2)/l_3**2)
            expr = expr_1 * expr_2 * expr_3

        return expr

class RQ(KernelBase):
    _name = 'RQ'

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
            alpha_1 = Constant('alpha_1')
            l_1 = Constant('l_1')
            theta_1 = Constant('theta_1')
            xi,xj = args
            expr = theta_1 * (1 + (xi - xj)**2/(2*alpha_1*l_1**2))**(-alpha_1)

        elif ldim == 2:
            alpha_1 = Constant('alpha_1')
            alpha_2 = Constant('alpha_2')
            l_1 = Constant('l_1')
            l_2 = Constant('l_2')
            theta_1 = Constant('theta_1')
            theta_2 = Constant('theta_2')
            xi,yi = args[0]
            xj,yj = args[1]
            expr_1 = theta_1 * (1 + (xi - xj)**2/(2*alpha_1*l_1**2))**(-alpha_1)
            expr_2 = theta_2 * (1 + (yi - yj)**2/(2*alpha_2*l_2**2))**(-alpha_2)
            expr = expr_1 * expr_2

        elif ldim == 3:
            alpha_1 = Constant('alpha_1')
            alpha_2 = Constant('alpha_2')
            alpha_3 = Constant('alpha_3')
            l_1 = Constant('l_1')
            l_2 = Constant('l_2')
            l_3 = Constant('l_3')
            theta_1 = Constant('theta_1')
            theta_2 = Constant('theta_2')
            theta_3 = Constant('theta_3')
            xi,yi,zi = args[0]
            xj,yj,zj = args[1]
            expr_1 = theta_1 * (1 + (xi - xj)**2/(2*alpha_1*l_1**2))**(-alpha_1)
            expr_2 = theta_2 * (1 + (yi - yj)**2/(2*alpha_2*l_2**2))**(-alpha_2)
            expr_3 = theta_3 * (1 + (zi - zj)**2/(2*alpha_3*l_3**2))**(-alpha_3)
            expr = expr_1 * expr_2 * expr_3

        return expr

class Periodic(KernelBase):
    _name = 'Periodic'

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
            p_1 = Constant('p_1')
            l_1 = Constant('l_1')
            theta_1 = Constant('theta_1')
            xi,xj = args
            expr = theta_1 * exp(-2*(sin(pi*(xi - xj)/p_1))**2/l_1**2)

        elif ldim == 2:
            p_1 = Constant('p_1')
            p_2 = Constant('p_2')
            l_1 = Constant('l_1')
            l_2 = Constant('l_2')
            xi,yi = args[0]
            xj,yj = args[1]
            theta_1 = Constant('theta_1')
            theta_2 = Constant('theta_2')
            expr_1 = theta_1 * exp(-2*(sin(pi*(xi - xj)/p_1))**2/l_1**2)
            expr_2 = theta_2 * exp(-2*(sin(pi*(yi - yj)/p_2))**2/l_2**2)
            expr = expr_1 * expr_2

        elif ldim == 3:
            p_1 = Constant('p_1')
            p_2 = Constant('p_2')
            p_3 = Constant('p_3')
            l_1 = Constant('l_1')
            l_2 = Constant('l_2')
            l_3 = Constant('l_3')
            theta_1 = Constant('theta_1')
            theta_2 = Constant('theta_2')
            theta_3 = Constant('theta_3')
            xi,yi,zi = args[0]
            xj,yj,zj = args[1]
            expr_1 = theta_1 * exp(-2*(sin(pi*(xi - xj)/p_1))**2/l_1**2)
            expr_2 = theta_2 * exp(-2*(sin(pi*(yi - yj)/p_2))**2/l_2**2)
            expr_3 = theta_3 * exp(-2*(sin(pi*(zi - zj)/p_3))**2/l_3**2)
            expr = expr_1 * expr_2 * expr_3

        return expr

class Linear(KernelBase):
    _name = 'Linear'

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

        theta_b = Constant('theta_b')
        theta_v = Constant('theta_v')
        if ldim == 1:
            c_1 = Constant('c_1')
            xi,xj = args
            expr = theta_b + theta_v*(xi-c_1)*(xj-c_1)

        elif ldim == 2:
            c_1 = Constant('c_1')
            c_2 = Constant('c_2')
            xi,yi = args[0]
            xj,yj = args[1]
            expr = theta_b + theta_v*((xi-c_1)*(xj-c_1) +
                                      (yi-c_2)*(yj-c_2))

        elif ldim == 3:
            c_1 = Constant('c_1')
            c_2 = Constant('c_2')
            c_3 = Constant('c_3')
            xi,yi,zi = args[0]
            xj,yj,zj = args[1]
            expr = theta_b + theta_v*((xi-c_1)*(xj-c_1) +
                                      (yi-c_2)*(yj-c_2) +
                                      (zi-c_3)*(zj-c_3))

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

    elif isinstance(L, Function):
        args = list(L.args)
        args = Tuple(*args)
        for _x, _xi in zip(x, xi):
            args = args.subs(_x, _xi)
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

    elif isinstance(expr, Symbol):
        coords = ['x', 'y', 'z']
        coords = [Symbol(i) for i in coords]
        if expr in coords:
            i = coords.index(expr)
            return xi[i]

        else:
            return expr

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

        # ...
        a = expr/Symbol('RBF')
        a = simplify(a)
        expr = a * Symbol('RBF')
        # ...

        expr = expr.subs(Symbol('RBF'), RBF(*variables))

    return expr

_template_1d = """
def {__KERNEL_NAME__}(x1, x2, {__ARGS__}):
{__NUMPY_IMPORT__}
    from numpy import zeros
    k = zeros((x1.size, x2.size))
    for i in range(x1.size):
        xi = x1[i]
        for j in range(x2.size):
            xj = x2[j]
            k[i,j] = {__EXPR__}
    return k
"""

_template_2d = """
def {__KERNEL_NAME__}(x1, x2, {__ARGS__}):
{__NUMPY_IMPORT__}
    from numpy import zeros
    k = zeros((x1.shape[0], x2.shape[0]))
    for i in range(x1.shape[0]):
        xi = x1[i, 0] ; yi = x1[i, 1]
        for j in range(x2.shape[0]):
            xj = x2[j, 0] ; yj = x2[j, 1]
            k[i,j] = {__EXPR__}
    return k
"""

_template_3d = """
def {__KERNEL_NAME__}(x1, x2, {__ARGS__}):
{__NUMPY_IMPORT__}
    from numpy import zeros
    k = zeros((x1.shape[0], x2.shape[0]))
    for i in range(x1.shape[0]):
        xi = x1[i, 0] ; yi = x1[i, 1] ; zi = x1[i, 2]
        for j in range(x2.shape[0]):
            xj = x2[j, 0] ; yj = x2[j, 1] ; zj = x2[j, 2]
            k[i,j] = {__EXPR__}
    return k
"""

def compile_kernels(expr, u, kernel, namespace=globals()):
    ldim = u.ldim
    if ldim == 1:
        xi = Symbol('xi')
        xj = Symbol('xj')

        Xi = [xi] ; Xj = [xj]

    elif ldim == 2:
        xi = Symbol('xi')
        yi = Symbol('yi')
        xj = Symbol('xj')
        yj = Symbol('yj')

        Xi = [xi,yi] ; Xj = [xj,yj]

    elif ldim == 3:
        xi = Symbol('xi')
        yi = Symbol('yi')
        zi = Symbol('zi')
        xj = Symbol('xj')
        yj = Symbol('yj')
        zj = Symbol('zj')

        Xi = [xi,yi,zi] ; Xj = [xj,yj,zj]

    positions = Xi + Xj

    if ldim == 1:
        kuu = kernel(xi, xj)

        K = evaluate(expr, u, Kernel('K'), xi)
        kfu = update_kernel(K, kernel, (xi, xj))

        K = evaluate(expr, u, Kernel('K'), xj)
        kuf = update_kernel(K, kernel, (xi, xj))

        K = evaluate(expr, u, Kernel('K'), (xi, xj))
        kff = update_kernel(K, kernel, (xi, xj))

    elif ldim == 2:
        kuu = kernel(Tuple(xi,yi), Tuple(xj,yj))

        K = evaluate(expr, u, Kernel('K'), (Tuple(xi, yi)))
        kfu = update_kernel(K, kernel, ((xi,yi), (xj,yj)))

        K = evaluate(expr, u, Kernel('K'), (Tuple(xj, yj)))
        kuf = update_kernel(K, kernel, ((xi,yi), (xj,yj)))

        K = evaluate(expr, u, Kernel('K'), (Tuple(xi,yi), Tuple(xj,yj)))
        kff = update_kernel(K, kernel, ((xi,yi), (xj,yj)))

    elif ldim == 3:
        kuu = kernel(Tuple(xi,yi,zi), Tuple(xj,yj,zj))

        K = evaluate(expr, u, Kernel('K'), (Tuple(xi, yi, zi)))
        kfu = update_kernel(K, kernel, ((xi,yi,zi), (xj,yj,zj)))

        K = evaluate(expr, u, Kernel('K'), (Tuple(xj, yj, zj)))
        kuf = update_kernel(K, kernel, ((xi,yi,zi), (xj,yj,zj)))

        K = evaluate(expr, u, Kernel('K'), (Tuple(xi,yi,zi), Tuple(xj,yj,zj)))
        kff = update_kernel(K, kernel, ((xi,yi,zi), (xj,yj,zj)))

    # ...
    d_k = {}
    d_k['kuu'] = kuu
    d_k['kfu'] = kfu
    d_k['kuf'] = kuf
    d_k['kff'] = kff
    # ...

    # ...
    if ldim == 1:
        template = _template_1d

    elif ldim == 2:
        template = _template_2d

    elif ldim == 3:
        template = _template_3d

    else:
        raise NotImplementedError('')
    # ...

    d_fct = {}
    d_args = {}
    for kernel_name, k in list(d_k.items()):

        args = [i for i in k.free_symbols if not(i in positions)]
        args = asarray(['{}'.format(i.name) for i in args])
        args.sort()
        args_str = ', '.join(i for i in args)

        expr = k

        # ... check if we find math functions in the expression
        math_functions = [str(type(i)) for i in preorder_traversal(expr) if isinstance(i, Function)]
        math_functions = [i for i in math_functions if i in known_math_functions]
        math_functions = list(set(math_functions)) # remove redundancies

        math_constants = [str(i) for i in preorder_traversal(expr) if isinstance(i, NumberSymbol)]
        math_constants = [i for i in math_constants if i in known_math_constants]
        math_constants = list(set(math_constants)) # remove redundancies

        tab = ' '*4
        numpy_import_str = print_import_from_numpy(math_functions+math_constants, tab)
        # ...

        code = template.format(__KERNEL_NAME__=kernel_name,
                               __ARGS__=args_str,
                               __NUMPY_IMPORT__=numpy_import_str,
                               __EXPR__=expr)

        # ...
#        print(code)
        exec(code, namespace)
        kernel = namespace[kernel_name]
        # ...

        d_fct[kernel_name] = kernel
        d_args[kernel_name] = args

    return d_fct, d_args


_template_nlml = """
def nlml_func(params, x1, x2, y1, y2, s_u, s_f, settings):
    import numpy as np
    params = np.exp(params)
{__ASSIGN_ARGS__}

    K = np.block([
        [
            {__KUU__}(x1, x2, {__KUU_ARGS__}) + s_u*np.identity(x1.shape[0]),
            {__KUF__}(x1, x2, {__KUF_ARGS__})
        ],
        [
            {__KFU__}(x1, x2, {__KFU_ARGS__}),
            {__KFF__}(x2, x2, {__KFF_ARGS__}) + s_f*np.identity(x2.shape[0])
        ]
    ])
    y = np.concatenate((y1, y2))
    val = 0.5*(np.log(abs(np.linalg.det(K))) + np.mat(y) * np.linalg.inv(K) * np.mat(y).T)
    return val.item(0)
"""

_template_nlml_svd = """
def nlml_func(params, x1, x2, y1, y2, s_u, s_f, settings):
    import numpy as np
    params = np.exp(params)
{__ASSIGN_ARGS__}

    K = np.block([
        [
            {__KUU__}(x1, x2, {__KUU_ARGS__}) + s_u*np.identity(x1.shape[0]),
            {__KUF__}(x1, x2, {__KUF_ARGS__})
        ],
        [
            {__KFU__}(x1, x2, {__KFU_ARGS__}),
            {__KFF__}(x2, x2, {__KFF_ARGS__}) + s_f*np.identity(x2.shape[0])
        ]
    ])
    y = np.concatenate((y1, y2))

    u, s, vh = np.linalg.svd(K)
#    print(s.min(), s.max())

    # ... using cumulative explained var
#    threshold = 0.9
#    s_sum = s.sum()
#    sn = []
#    ti = 0
#    i = 0
#    while ti < threshold:
#        sn.append(s[i])
#        ti += s[i]/s_sum
#        i += 1
    # ...

    # ... using min var
    s_max = s.max()
#    sn = [i for i in s if np.abs(i) > 0.005]
    sn = [i for i in s if np.abs(i)/s_max > 1.e-3]
    # ...

    debug = False
    if 'debug' in list(settings.keys()):
        debug = True

    if debug:
        if 'iteration' in list(settings.keys()):
            iteration = settings['iteration']

            label = 'iteration ' + str(iteration)
            plt.plot(np.log(s), '-', label=label)

            settings['iteration'] += 1

    n = len(sn)
#    print('> n = ', n, '/', len(y))
    un = u[:,:n]
    vhn = vh[:n,:]
    sn = s[:n]

    d = sn.prod()
    zl = np.dot(vhn, y)
    zr = np.dot(y, un)
    zl = np.conjugate(zl)
    zr = np.conjugate(zr)

    val = np.log(abs(d))
    for i in range(0, n):
        val += zl[i]*zr[i]/sn[i]

    return val
"""

class NLML(object):
    def __init__(self, expr, u, kernel, namespace=globals(), **settings):

        if isinstance(kernel, str):
            kernel = eval(kernel)

        self._func = self._compile_nlml(expr, u, kernel, namespace=namespace)

        self._debug = settings.pop('debug', False)
        if self.debug:
            settings['iteration'] = 0
            settings['debug'] = self.debug

        self._settings = settings

    @property
    def func(self):
        return self._func

    @property
    def args(self):
        return self._args

    @property
    def params(self):
        return self._params

    @property
    def iteration(self):
        return self._iteration

    @property
    def debug(self):
        return self._debug

    @property
    def settings(self):
        return self._settings

    @property
    def x_u(self):
        return self._x_u

    @property
    def x_f(self):
        return self._x_f

    @property
    def u(self):
        return self._u

    @property
    def f(self):
        return self._f

    def set_u(self, x_u, u):
        self._x_u = x_u
        self._u   = u

    def set_f(self, x_f, f):
        self._x_f = x_f
        self._f   = f

    def map_args(self, values):
        d = {}
        for n,v in zip(self.args, values):
            d[n] = v
        return d

    def _compile_nlml(self, expr, u, kernel, namespace=globals()):
        d_fct, d_args = compile_kernels(expr, u, kernel)
        d_args_str = {}
        for name, args in list(d_args.items()):
            d_args_str[name] = ', '.join(i for i in args)

        args = []
        for name, values in list(d_args.items()):
            args += [str(i) for i in values]

        args = unique(args)
        args.sort()

        # set args
        self._args = args

        # parametric variables
        params = [i.name for i in expr.free_symbols if i.name in args]
        self._params = params

        stmts = []
        for i, a in enumerate(args):
            pattern = '{__ARG__} = params[{i}]'
            stmt = pattern.format(__ARG__=a, i=i)
            stmts.append(stmt)

        tab = ' '*4
        assign_args_str = '\n'.join(tab + i for i in stmts)

        # ...
#        template = _template_nlml
        template = _template_nlml_svd
        # ...

        code = template.format(__KUU__='kuu', __KUU_ARGS__=d_args_str['kuu'],
                               __KFU__='kfu', __KFU_ARGS__=d_args_str['kfu'],
                               __KUF__='kuf', __KUF_ARGS__=d_args_str['kuf'],
                               __KFF__='kff', __KFF_ARGS__=d_args_str['kff'],
                               __ASSIGN_ARGS__=assign_args_str)

        # ...
#        print(code)
#        import sys; sys.exit(0)
        exec(code, namespace)
        nlml = namespace['nlml_func']
        # ...

        return nlml

    def __call__(self, args, x_u=None, x_f=None, u=None, f=None, s_u=1.e-6,
                 s_f=1.e-6):
        if x_u is None:
            x_u = self.x_u

        if x_f is None:
            x_f = self.x_f

        if u is None:
            u = self.u

        if f is None:
            f = self.f

        return self.func(args, x_u, x_f, u, f, s_u, s_f, self.settings)
