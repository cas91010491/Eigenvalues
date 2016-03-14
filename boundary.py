"""
Provides classes of boundary conditons and other related tools.

Supports mixed boundary condition given by (in)equalities.
"""

# pylama:ignore=E731
import re
from dolfin import FacetFunction, SubDomain
from paramchecker import evaluate
import numpy as np
from cPickle import dumps

bcApplyChoices = ['global', 'conditions']  # no sides yet


class GenericBoundary(object):

    """ Generic boundary conditon class. """

    value = 0

    def __init__(self):
        """ Set all default condition values. """
        self.param = "None"
        self.applyTo = 'global'
        self.conditions = 'True'
        self.parValue = None
        self.check = "lambda x, y, z=0: True"
        self.on = True

    def __repr__(self):
        """Return string representation suitable for object reconstruction."""
        s = self.name
        if self.param != "None":
            s += ' with parameter '+self.param
        s += '; '+self.applyTo
        if self.applyTo != "global":
            s += ': '+self.conditions
        return s

    def update(self, applyTo='global', conditions='True', param="None"):
        """
        Parse boundary condition parameters.

        applyTo sets the type of the boundary specification:
            sides: list of boundary sides/triangles to which BC should apply
            conditions: logical conditions for (x,y,z)
                OR is placed between elements if a list is provided
            global: apply to the whole boundary
        param is used if a value is associated with a boundary condition.
        self.on =
            True: forces BC to apply only on the mesh boundary
            False: interior points can be a part of BC
        """
        self.on = True
        applyTo = applyTo.lower()
        self.applyTo = applyTo
        if applyTo == 'sides':
            self.initializeSides(conditions)
        elif applyTo == 'conditions':
            self.initializeConditions(conditions)
        else:  # global
            self.applyTo = "global"
        try:
            self.parValue = evaluate(param)[0]
            self.param = param
        except:
            pass

    @staticmethod
    def preprocessConditions(conditions):
        """ Turn a list of conditions into a function. """
        conditions = re.sub(r'&+', ' and ', conditions)
        conditions = re.sub(r'\|+', ' or ', conditions)
        conditions = re.sub(r'==+', '=', conditions)
        conditions = re.sub(r'(?<![<>])=', '==', conditions)
        conditions = "lambda x, y, z=0: any([ " + conditions + " ])"
        return conditions

    def initializeConditions(self, conditions):
        """ Try to parse logical conditions given by the user. """
        try:
            new = self.preprocessConditions(conditions)
            evaluate(new)  # check if valid
            self.conditions = conditions
            self.check = new
        except:
            try:
                new = self.preprocessConditions(self.conditions)
                evaluate(new)
                self.check = new
            except:
                self.conditions = "True"
                self.check = "lambda x, y, z=0: True"

    def getTest(self):
        """ Return a function which can check if a point belongs to bc. """
        if self.applyTo == 'global':
            return lambda x, on: on
        fun = evaluate(self.check)[0]
        if self.on:
            return lambda x, on: fun(*x) and on
        else:
            return lambda x, on: fun(*x)

    def initializeSides(self, sides):
        """
        Generate point in interval/triangle tests for all boundary cells.

        Make self.check functions based on boundary cells from sides.

        Not implemented yet.
        """
        self.conditions = "All"
        self.check = "lambda x, y, z=0: True"


class NeumannBoundary(GenericBoundary):

    """
    Neumann.

    Neumann boundary condition.

    Since this is the natural BC, it essentialy removes other conditions.
    """

    value = 0

    def update(self, applyTo='global', conditions='True', param="None"):
        """Neumann does not use parameters."""
        super(NeumannBoundary, self).update(applyTo, conditions, "None")


class DirichletBoundary(GenericBoundary):

    """
    Dirichlet.

    Dirichlet boundary condition.

    Can also be applied in the interior of the domain.
    """

    value = 1

    def update(self, applyTo='global', conditions='True', param="None"):
        """Dirichlet conditon can be applied to the interior points."""
        if applyTo == 'inside':
            super(DirichletBoundary, self).update('conditions',
                                                  conditions, "None")
            self.on = False
            self.applyTo = 'inside'
        else:
            super(DirichletBoundary, self).update(applyTo, conditions, "None")


class RobinBoundary(GenericBoundary):

    """
    Robin.

    Robin boundary condition.

    The default value of the Robin parameter is 1.
    """

    value = 2

    def __init__(self):
        """Consecutive Robin objects must be distinguishable."""
        super(RobinBoundary, self).__init__()
        self.value = RobinBoundary.value
        RobinBoundary.value += 1
        self.update(param="1")


class SteklovBoundary(GenericBoundary):

    r"""
    Steklov.

    Steklov boundary condition.

    The default value of the Steklov parameter is 1.

    If any part of the boundary has a Steklov condition applied, \
    the denominator of the Rayleigh quotient is evaluated over Steklov \
    boundary instead of inside the domain.
    """

    value = -1

    def __init__(self):
        """Consecutive Steklov objects must be distinguishable."""
        super(SteklovBoundary, self).__init__()
        self.value = SteklovBoundary.value
        SteklovBoundary.value -= 1
        self.update(param="1")


#
# tools
#

def build_bc_dict():
    """
    Return dictionary of all boundary conditions.

    Keys:   BC names
    Values: BC classes
    """
    from inspect import getdoc
    import re
    bc_dict = {}
    for subcls in GenericBoundary.__subclasses__():
        lines = re.sub(r'\\\n', '', getdoc(subcls)).split("\n")
        subcls.name = lines[0][:-1]
        subcls.help = "\n".join(lines[2:])
        try:
            dumps(subcls())
            bc_dict[subcls.name] = subcls
        except:
            print subcls.name + ' not picklable'
    return bc_dict


class OnBoundary(SubDomain):

    """ Boundary as subdomain. """

    def inside(self, _, on):
        """ True for any boundary point. """
        return on


def marked_boundary(mesh):
    """ Return array of vertex values with 1 on boundary, 0 otherwise. """
    fun = FacetFunction("int", mesh, 0)
    on = OnBoundary()
    on.mark(fun, 1)
    return fun.array()


def mark_conditions(mesh, lst):
    """ Mark all boundary conditions from the list. """
    fun = FacetFunction("int", mesh, 0)
    for bc in lst:
        sub = OnBoundary()
        # overwrite inside function with the one from bc
        sub.inside = bc.getTest()
        sub.mark(fun, bc.value)
    return fun.array()


def get_bc_parts(mesh, lst):
    """
    Build a size_t function with boundary condition parts.

    Returns all Robin and Steklov conditions that need to be applied,
    the shift needed to get a nonnegative function,
    and the function itself.
    """
    if len(lst) > 0:
        shift = max(0, -min(e.value for e in lst))
    else:
        return [], [], 0, FacetFunction("size_t", mesh, 0)
    # values must be shifted by smallest Steklov value since size_t is unsigned
    fun = FacetFunction("size_t", mesh, shift)
    for bc in lst:
        sub = OnBoundary()
        # overwrite inside function with the one from bc
        sub.inside = bc.getTest()
        sub.mark(fun, bc.value + shift)
    # some conditions may cancel eachother
    exist = set(np.unique(fun.array()))
    lst = [e for e in lst if e.value+shift in exist]
    # separate Robin and Steklov, Dirichlet and Neumann are irrelevant
    Robin = [e for e in lst if e.value > 1 and e.parValue != 0]
    Steklov = [e for e in lst if e.value < 0 and e.parValue != 0]
    return Robin, Steklov, shift, fun
