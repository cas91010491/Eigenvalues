""" Class and tools for handling user supplied parameters. """
from __future__ import division

import re
import numpy as np
import collections
from math import pi
from numpy.random import rand, seed
from dolfin.cpp.function import near

np.seterr(divide='ignore', invalid='ignore')


def flatten(lst):
    """ Flatten any iterable. """
    if isinstance(lst, collections.Iterable) and \
            not isinstance(lst, basestring):
        return [a for i in lst for a in flatten(i)]
    else:
        return [lst]


def clip(lst, type_, lower=-np.inf, upper=np.inf):
    r"""
    Each element in lst should be of appropriate type and in the interval.

    Bounds can be lists at least as long as lst.
    lst is also flattened, or made into a one element list
    """
    if not isinstance(lst, collections.Iterable):
        lst = [lst]
    if isinstance(lower, collections.Iterable):
        lower = lower[:len(lst)]
    if isinstance(upper, collections.Iterable):
        upper = upper[:len(lst)]
    try:
        lst = np.clip(lst, lower, upper)
        return [type_(e) for e in lst]
    except:
        return []


# pylama:ignore=E731
def evaluate(s, subs={}):
    """
    Somewhat safer eval with embedded numpy/math functions.

    Always returns a list. Also handles space separated numbers.
    """
    ns = vars(np).copy()
    try:
        ns.update(subs)
    except:
        pass
    seed()
    ns['__builtins__'] = None
    ns['randA'] = lambda n: sorted(rand(n)*360)
    ns['randD'] = lambda n, min=0.5: rand(n)+min
    ns['rand'] = rand
    ns['randU'] = lambda n: np.array(
        (lambda x: (np.cos(x), np.sin(x)))(rand(n)*2*pi)).transpose()
    ns['norm'] = np.absolute
    ns['abs'] = np.absolute
    ns['arg'] = np.angle
    ns['re'] = np.real
    ns['im'] = np.imag
    ns['range'] = range
    ns['near'] = near
    ns['False'] = False
    ns['True'] = True
    s = re.sub(r"\.(?![0-9])", "", s)
    s = re.sub(r"\^", "**", s)
    try:
        ans = eval(s, ns)
    except:
        s = re.sub(r"\s+", ",", s)
        try:
            ans = eval(s, ns)
        except:
            ans = []
    if not hasattr(ans, "__len__") or isinstance(ans, basestring):
        ans = [ans]
    return ans


class ParamChecker(object):

    """
    Class for handling user supplied parameters with default value.

    Partial input is also supported via padding to full format.
    Stores parsed params in values.

    Empty params force use of default values.
    """

    default = ""
    full = None
    def_pref = ""

    def __init__(self):
        """Set initial parameter values."""
        self.params = self.default
        self.prefix = self.def_pref
        self.values = []

    @staticmethod
    def format(lst):
        r"""
        Ensure list has appropriate format. Always return list.

        Use list[1:2] instead of list[1] to make the function independent of \
        the list length.
        """
        return lst

    def eval(self, params, prefix=''):
        """
        Turn params string into format fitting list of values.

        If params cannot be evaluate, use self.params.
        If params are '', use self.default.
        If params can be evaluated, update self.params.

        Prefix is prepended to params before evaluation, and has it's own
        default value def_pref.
        """
        # logic is too complicated here
        try:
            self.values = self.format(evaluate(prefix + params))
            if params:
                self.params = params
                self.prefix = prefix
            else:
                # this should never happen
                assert False
                self.params = self.default
                self.prefix = self.def_pref
        except:
            if not params:
                self.params = self.default
                self.prefix = self.def_pref
            self.values = self.format(evaluate(self.prefix + self.params))
        return self.values

    def pad(self, lst):
        """
        Extend x using self.full.

        Allows for default parameter values.
        """
        temp = list(self.full)
        temp[:len(lst)] = lst[:len(temp)]
        return tuple(temp)
