# Fork of https://github.com/patrick-kidger/sympytorch

import collections as co
import functools as ft

import numpy as np  # noqa: F401
import sympy  # type: ignore
from sympy.codegen.cfunctions import log2, log10  # type: ignore


def _reduce(fn):
    def fn_(*args):
        return ft.reduce(fn, args)

    return fn_


torch_initialized = False
torch = None
SingleSymPyModule = None


def _initialize_torch():
    global torch_initialized
    global torch
    global SingleSymPyModule

    # Way to lazy load torch, only if this is called,
    # but still allow this module to be loaded in __init__
    if not torch_initialized:
        import torch as _torch

        torch = _torch

        # Allows PyTorch to map Piecewise functions:
        def expr_cond_pair(expr, cond):
            if isinstance(cond, torch.Tensor) and not isinstance(expr, torch.Tensor):
                expr = torch.tensor(expr, dtype=cond.dtype, device=cond.device)
            elif isinstance(expr, torch.Tensor) and not isinstance(cond, torch.Tensor):
                cond = torch.tensor(cond, dtype=expr.dtype, device=expr.device)
            else:
                return expr, cond

            # First, make sure expr and cond are same size:
            if expr.shape != cond.shape:
                if len(expr.shape) == 0:
                    expr = expr.expand(cond.shape)
                elif len(cond.shape) == 0:
                    cond = cond.expand(expr.shape)
                else:
                    raise ValueError(
                        "expr and cond must have same shape, or one must be a scalar."
                    )
            return expr, cond

        def if_then_else(*conds):
            a, b, c = conds
            return torch.where(
                a, torch.where(b, True, False), torch.where(c, True, False)
            )

        def piecewise(*expr_conds):
            output = None
            already_used = None
            for expr, cond in expr_conds:
                if not isinstance(cond, torch.Tensor) and not isinstance(
                    expr, torch.Tensor
                ):
                    # When we just have scalars, have to do this a bit more complicated
                    # due to the fact that we need to evaluate on the correct device.
                    if output is None:
                        already_used = cond
                        output = expr if cond else 0.0
                    else:
                        if not isinstance(output, torch.Tensor):
                            output += expr if cond and not already_used else 0.0
                            already_used = already_used or cond
                        else:
                            expr = torch.tensor(
                                expr, dtype=output.dtype, device=output.device
                            ).expand(output.shape)
                            output += torch.where(
                                cond & ~already_used, expr, torch.zeros_like(expr)
                            )
                            already_used = already_used | cond
                else:
                    if output is None:
                        already_used = cond
                        output = torch.where(cond, expr, torch.zeros_like(expr))
                    else:
                        output += torch.where(
                            cond.bool() & ~already_used, expr, torch.zeros_like(expr)
                        )
                        already_used = already_used | cond.bool()
            return output

        # TODO: Add test that makes sure tensors are on the same device

        _global_func_lookup = {
            sympy.Mul: _reduce(torch.mul),
            sympy.Add: _reduce(torch.add),
            sympy.div: torch.div,
            sympy.Abs: torch.abs,
            sympy.sign: torch.sign,
            # Note: May raise error for ints.
            sympy.ceiling: torch.ceil,
            sympy.floor: torch.floor,
            sympy.log: torch.log,
            log2: torch.log2,
            log10: torch.log10,
            sympy.exp: torch.exp,
            sympy.sqrt: torch.sqrt,
            sympy.cos: torch.cos,
            sympy.acos: torch.acos,
            sympy.sin: torch.sin,
            sympy.asin: torch.asin,
            sympy.tan: torch.tan,
            sympy.atan: torch.atan,
            sympy.atan2: torch.atan2,
            # Note: May give NaN for complex results.
            sympy.cosh: torch.cosh,
            sympy.acosh: torch.acosh,
            sympy.sinh: torch.sinh,
            sympy.asinh: torch.asinh,
            sympy.tanh: torch.tanh,
            sympy.atanh: torch.atanh,
            sympy.Pow: torch.pow,
            sympy.re: torch.real,
            sympy.im: torch.imag,
            sympy.arg: torch.angle,
            # Note: May raise error for ints and complexes
            sympy.erf: torch.erf,
            sympy.loggamma: torch.lgamma,
            sympy.Eq: torch.eq,
            sympy.Ne: torch.ne,
            sympy.StrictGreaterThan: torch.gt,
            sympy.StrictLessThan: torch.lt,
            sympy.LessThan: torch.le,
            sympy.GreaterThan: torch.ge,
            sympy.And: torch.logical_and,
            sympy.Or: torch.logical_or,
            sympy.Not: torch.logical_not,
            sympy.Max: torch.max,
            sympy.Min: torch.min,
            sympy.Mod: torch.remainder,
            sympy.Heaviside: torch.heaviside,
            sympy.core.numbers.Half: (lambda: 0.5),
            sympy.core.numbers.One: (lambda: 1.0),
            sympy.logic.boolalg.Boolean: lambda x: x,
            sympy.logic.boolalg.BooleanTrue: (lambda: True),
            sympy.logic.boolalg.BooleanFalse: (lambda: False),
            sympy.functions.elementary.piecewise.ExprCondPair: expr_cond_pair,
            sympy.Piecewise: piecewise,
            sympy.logic.boolalg.ITE: if_then_else,
        }

        class _Node(torch.nn.Module):
            """Forked from https://github.com/patrick-kidger/sympytorch"""

            def __init__(self, *, expr, _memodict, _func_lookup, **kwargs):
                super().__init__(**kwargs)

                self._sympy_func = expr.func

                if issubclass(expr.func, sympy.Float):
                    self._value = torch.nn.Parameter(torch.tensor(float(expr)))
                    self._torch_func = lambda: self._value
                    self._args = ()
                elif issubclass(expr.func, sympy.Rational):
                    # Includes Integer, since Integer is a subclass of Rational
                    self.register_buffer("_value", torch.tensor(float(expr)))
                    self._torch_func = lambda: self._value
                    self._args = ()
                elif issubclass(expr.func, sympy.UnevaluatedExpr):
                    if len(expr.args) != 1 or not issubclass(
                        expr.args[0].func, sympy.Float
                    ):
                        raise ValueError(
                            "UnevaluatedExpr should only be used to wrap floats."
                        )
                    self.register_buffer("_value", torch.tensor(float(expr.args[0])))
                    self._torch_func = lambda: self._value
                    self._args = ()
                elif issubclass(expr.func, sympy.NumberSymbol):
                    # Can get here from exp(1) or exact pi
                    self.register_buffer("_value", torch.tensor(float(expr)))
                    self._torch_func = lambda: self._value
                    self._args = ()
                elif issubclass(expr.func, sympy.Symbol):
                    self._name = expr.name
                    self._torch_func = lambda value: value
                    self._args = ((lambda memodict: memodict[expr.name]),)
                else:
                    try:
                        self._torch_func = _func_lookup[expr.func]
                    except KeyError:
                        raise KeyError(
                            f"Function {expr.func} was not found in Torch function mappings. "
                            "Please add it to extra_torch_mappings in the format, e.g., "
                            "{sympy.sqrt: torch.sqrt}."
                        )
                    args = []
                    for arg in expr.args:
                        try:
                            arg_ = _memodict[arg]
                        except KeyError:
                            arg_ = type(self)(
                                expr=arg,
                                _memodict=_memodict,
                                _func_lookup=_func_lookup,
                                **kwargs,
                            )
                            _memodict[arg] = arg_
                        args.append(arg_)
                    self._args = torch.nn.ModuleList(args)

            def forward(self, memodict):
                args = []
                for arg in self._args:
                    try:
                        arg_ = memodict[arg]
                    except KeyError:
                        arg_ = arg(memodict)
                        memodict[arg] = arg_
                    args.append(arg_)
                try:
                    return self._torch_func(*args)
                except Exception as err:
                    # Add information about the current node to the error:
                    raise type(err)(
                        f"Error occurred in node {self._sympy_func} with args {args}"
                    )

        class _SingleSymPyModule(torch.nn.Module):
            """Forked from https://github.com/patrick-kidger/sympytorch"""

            def __init__(
                self, expression, symbols_in, selection=None, extra_funcs=None, **kwargs
            ):
                super().__init__(**kwargs)

                if extra_funcs is None:
                    extra_funcs = {}
                _func_lookup = co.ChainMap(_global_func_lookup, extra_funcs)

                _memodict = {}
                self._node = _Node(
                    expr=expression, _memodict=_memodict, _func_lookup=_func_lookup
                )
                self._expression_string = str(expression)
                self._selection = selection
                self.symbols_in = [str(symbol) for symbol in symbols_in]

            def __repr__(self):
                return f"{type(self).__name__}(expression={self._expression_string})"

            def forward(self, X):
                if self._selection is not None:
                    X = X[:, self._selection]
                symbols = {symbol: X[:, i] for i, symbol in enumerate(self.symbols_in)}
                return self._node(symbols)

        SingleSymPyModule = _SingleSymPyModule


def sympy2torch(expression, symbols_in, selection=None, extra_torch_mappings=None):
    """Returns a module for a given sympy expression with trainable parameters;

    This function will assume the input to the module is a matrix X, where
        each column corresponds to each symbol you pass in `symbols_in`.
    """
    global SingleSymPyModule

    _initialize_torch()

    return SingleSymPyModule(
        expression, symbols_in, selection=selection, extra_funcs=extra_torch_mappings
    )
