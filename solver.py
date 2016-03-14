"""
Provides Solver class and other tools for solving eigenvalue problems.

Handles all adaptive cases.
"""

from dolfin import refine, CellFunction, Constant, Measure, FunctionSpace, \
    TrialFunction, TestFunction, dot, assemble, dx, DirichletBC, \
    grad, SLEPcEigenSolver, Function, FacetFunction, PETScMatrix, Mesh, \
    project, interpolate, ds, \
    Expression, DomainBoundary
from boundary import get_bc_parts
from transforms import transform_mesh
from domains import build_mesh
import numpy as np
import time

USE_EIGEN = False

# FIXME finish EIGEN implementation


class Solver:

    """ Main eigenvalue solver class. """

    def __init__(self, mesh, bcList, transformList, deg=2,
                 bcLast=False, method='CG',
                 wTop='1', wBottom='1'):
        """
        Initialize basic data.

        Method should be either CG or CR.
        """
        self.pickleMesh = None
        self.mesh = mesh
        self.deg = deg
        self.dim = mesh.topology().dim()
        self.size = self.mesh.size(self.dim)
        self.exit = False
        self.bcList = bcList
        self.transformList = transformList
        self.bcLast = bcLast
        if method in {'nonconforming', 'lower bound'}:
            self.method = 'CR'
            self.deg = 1
        else:
            self.method = 'CG'
        self.CGbound = (method == 'lower bound')
        self.monitor = None
        self.adaptive = self.upTo = self.edge = False
        self.number = 10
        self.target = None
        self.wTop = wTop
        self.wBottom = wBottom

    def refineTo(self, size, upTo=False, edge=False):
        """
        Save arguments.

        Procedure is done while solving.
        """
        self.upTo = upTo
        self.size = size
        self.edge = edge

    def refineMesh(self):
        """ Perform mesh refinement. """
        if self.upTo:
            mesh = refine_mesh_upto(self.mesh, self.size, self.edge)
        else:
            mesh = refine_mesh(self.mesh, self.size, self.edge)
        return mesh

    def __call__(self, monitor):
        """ Call solvers and return eigenvalues/eigenfunctions. """
        self.monitor = monitor
        self.mesh = build_mesh(*self.pickleMesh)
        results = list(self.solve())
        results.extend(self.getGeometry())
        return results

    def progress(self, s):
        """
        Send progress report.

        Assumes monitor is a queue (as in multiprocessing), or a function.
        """
        try:
            self.monitor.put(s)
        except:
            try:
                self.monitor(s)
            except:
                pass
        time.sleep(0.01)

    def newFunction(self):
        """ Create a function in the appropriate FEM space. """
        if not self.mesh:
            self.addMesh()
        return Function(FunctionSpace(self.mesh, 'CG', 1))

    def addMesh(self, mesh=None):
        """
        Keep fully transformed mesh.

        This breaks pickling.
        """
        if mesh is None:
            self.mesh = build_mesh(*self.pickleMesh)
            self.mesh = self.refineMesh()
            self.mesh = transform_mesh(self.mesh, self.transformList)
            self.finalsize = self.mesh.size(self.dim)
        else:
            self.mesh = mesh
        self.extraRefine = self.deg > 1
        if self.extraRefine:
            self.mesh = refine(self.mesh)

    def removeMesh(self):
        """ Remove mesh to restore pickling ability. """
        if self.pickleMesh is None:
            self.pickleMesh = [self.mesh.cells(), self.mesh.coordinates()]
        self.mesh = None

    def solveFor(self, number=10, target=None, exit=False):
        """ Save parameters related to number of eigenvalues. """
        self.number = number
        self.target = target
        self.exit = exit

    def solve(self):
        """ Find eigenvalues for transformed mesh. """
        self.progress("Building mesh.")
        # build transformed mesh
        mesh = self.refineMesh()
        # dim = mesh.topology().dim()
        if self.bcLast:
            mesh = transform_mesh(mesh, self.transformList)
            Robin, Steklov, shift, bcs = get_bc_parts(mesh, self.bcList)
        else:
            Robin, Steklov, shift, bcs = get_bc_parts(mesh, self.bcList)
            mesh = transform_mesh(mesh, self.transformList)
            # boundary conditions computed on non-transformed mesh
            # copy the values to transformed mesh
            fun = FacetFunction("size_t", mesh, shift)
            fun.array()[:] = bcs.array()[:]
            bcs = fun
        ds = Measure('ds', domain=mesh, subdomain_data=bcs)
        V = FunctionSpace(mesh, self.method, self.deg)
        u = TrialFunction(V)
        v = TestFunction(V)
        self.progress("Assembling matrices.")
        wTop = Expression(self.wTop)
        wBottom = Expression(self.wBottom)

        #
        # build stiffness matrix form
        #
        s = dot(grad(u), grad(v))*wTop*dx
        # add Robin parts
        for bc in Robin:
            s += Constant(bc.parValue)*u*v*wTop*ds(bc.value+shift)

        #
        # build mass matrix form
        #
        if len(Steklov) > 0:
            m = 0
            for bc in Steklov:
                m += Constant(bc.parValue)*u*v*wBottom*ds(bc.value+shift)
        else:
            m = u*v*wBottom*dx

        # assemble
        # if USE_EIGEN:
        #     S, M = EigenMatrix(), EigenMatrix()
            # tempv = EigenVector()
        # else:
        S, M = PETScMatrix(), PETScMatrix()
        # tempv = PETScVector()

        if not np.any(bcs.array() == shift+1):
            # no Dirichlet parts
            assemble(s, tensor=S)
            assemble(m, tensor=M)
        else:
            #
            # with EIGEN we could
            #   apply Dirichlet condition symmetrically
            #   completely remove rows and columns
            #
            # Dirichlet parts are marked with shift+1
            #
            # temp = Constant(0)*v*dx
            bc = DirichletBC(V, Constant(0.0), bcs, shift+1)
            # assemble_system(s, temp, bc, A_tensor=S, b_tensor=tempv)
            # assemble_system(m, temp, bc, A_tensor=M, b_tensor=tempv)
            assemble(s, tensor=S)
            bc.apply(S)
            assemble(m, tensor=M)
            # bc.zero(M)

        # if USE_EIGEN:
        #    M = M.sparray()
        #    M.eliminate_zeros()
        #    print M.shape
        #    indices = M.indptr[:-1] - M.indptr[1:] < 0
        #    M = M[indices, :].tocsc()[:, indices]
        #    S = S.sparray()[indices, :].tocsc()[:, indices]
        #    print M.shape
        #
        # solve the eigenvalue problem
        #
        self.progress("Solving eigenvalue problem.")
        eigensolver = SLEPcEigenSolver(S, M)
        eigensolver.parameters["problem_type"] = "gen_hermitian"
        eigensolver.parameters["solver"] = "krylov-schur"
        if self.target is not None:
            eigensolver.parameters["spectrum"] = "target real"
            eigensolver.parameters["spectral_shift"] = self.target
        else:
            eigensolver.parameters["spectrum"] = "smallest magnitude"
            eigensolver.parameters["spectral_shift"] = -0.01
        eigensolver.parameters["spectral_transform"] = "shift-and-invert"
        eigensolver.solve(self.number)
        self.progress("Generating eigenfunctions.")
        if eigensolver.get_number_converged() == 0:
            return None
        eigf = []
        eigv = []
        if self.deg > 1:
            mesh = refine(mesh)
        W = FunctionSpace(mesh, 'CG', 1)
        for i in range(eigensolver.get_number_converged()):
            pair = eigensolver.get_eigenpair(i)[::2]
            eigv.append(pair[0])
            u = Function(V)
            u.vector()[:] = pair[1]
            eigf.append(interpolate(u, W))
        return eigv, eigf

    def SolveExit(self):
        """ Find expected exit time/torsion function. """
        pass

    def getGeometry(self):
        """ Compute geometric factors. """
        self.progress("Computing geometric factors.")
        # build transformed mesh
        mesh = self.refineMesh()
        mesh = transform_mesh(mesh, self.transformList)
        V = FunctionSpace(mesh, 'CG', 1)
        u = Function(V)
        u.vector()[:] = 1
        # area/volume
        # weight from denominator of Rayleigh
        w = Expression(self.wBottom)
        geometry = {}
        A = geometry['A'] = assemble(u*w*dx)
        # perimeter/surface area
        geometry['P'] = assemble(u*w*ds)
        # center of mass
        x = Expression('x[0]')
        y = Expression('x[1]')
        cx = assemble(u*x*w*dx)/A
        cy = assemble(u*y*w*dx)/A
        c = [cx, cy]
        if self.dim == 3:
            z = Expression('x[2]')
            cz = assemble(u*z*w*dx)/A
            c.append(cz)
        geometry['c'] = c
        # moment of inertia
        if self.dim == 2:
            f = Expression(
                "(x[0]-cx)*(x[0]-cx)+(x[1]-cy)*(x[1]-cy)",
                cx=cx, cy=cy)
        else:
            f = Expression(
                "(x[0]-cx)*(x[0]-cx)+(x[1]-cy)*(x[1]-cy)+(x[2]-cz)*(x[2]-cz)",
                cx=cx, cy=cy, cz=cz)
        geometry['I'] = assemble(u*f*w*dx)
        # TODO: implement Gs
        # TODO: implement diameter and inradius
        geometry['D'] = None
        geometry['R'] = None
        return [geometry]

    def AdaptiveSolve(self):
        """ Adaptive refine and solve. """
        pass


def refine_mesh(mesh, size, edge=False):
    """ Refine mesh to at least given size, using one of two methods. """
    dim = mesh.topology().dim()
    if not edge:
        # FEniCS 1.5 and 1.6 have a bug which prevents uniform refinement
        while mesh.size(dim) < size:
            mesh = refine(mesh)
    else:
        # Refine based on MeshFunction
        while mesh.size(dim) < size:
            print refine(mesh).size(dim)
            full = CellFunction("bool", mesh, True)
            print refine(mesh, full).size(dim)
            mesh = refine(mesh, full)
    return mesh


def refine_mesh_upto(mesh, size, edge=False):
    """ Refine mesh to at most given size, using one of two methods. """
    dim = mesh.topology().dim()
    if mesh.size(dim) > size:
        return mesh
    if not edge:
        while True:
            # FEniCS 1.5 and 1.6 have a bug which prevents uniform refinement
            mesh2 = refine(mesh)
            if mesh2.size(dim) > size:
                return mesh
            mesh = mesh2
    else:
        # Refine based on MeshFunction
        while True:
            all = CellFunction("bool", mesh, True)
            mesh2 = refine(mesh, all)
            if mesh2.size(dim) > size:
                return mesh
            mesh = mesh2


def shiftMesh(mesh, vector):
    """ Shift mesh by vector. """
    mesh.coordinates()[:, :] += np.array(vector)[None, :]


def symmetrize(u, d, sym):
    """ Symmetrize function u. """
    if len(d) == 3:
        # three dimensions -> cycle XYZ
        return cyclic3D(u)
    elif len(d) >= 4:
        # four dimensions -> rotations in 2D
        return rotational(u, d[-1])
    nrm = np.linalg.norm(u.vector())
    V = u.function_space()
    mesh = Mesh(V.mesh())

    # test if domain is symmetric using function equal 0 inside, 1 on boundary
    # extrapolation will force large values if not symmetric since the flipped
    # domain is different
    bc = DirichletBC(V, 1, DomainBoundary())
    test = Function(V)
    bc.apply(test.vector())

    if len(d) == 2:
        # two dimensions given: swap dimensions
        mesh.coordinates()[:, d] = mesh.coordinates()[:, d[::-1]]
    else:
        # one dimension given: reflect
        mesh.coordinates()[:, d[0]] *= -1
    W = FunctionSpace(mesh, 'CG', 1)
    try:
        # testing
        test = interpolate(Function(W, test.vector()), V)
        # max-min should be around 1 if domain was symmetric
        # may be slightly above due to boundary approximation
        assert max(test.vector()) - min(test.vector()) < 1.1

        v = interpolate(Function(W, u.vector()), V)
        if sym:
            # symmetric
            pr = project(u+v)
        else:
            # antisymmetric
            pr = project(u-v)
        # small solution norm most likely means that symmetrization gives
        # trivial function
        assert np.linalg.norm(pr.vector())/nrm > 0.01
        return pr
    except:
        # symmetrization failed for some reason
        print "Symmetrization " + str(d) + " failed!"
        return u


def cyclic3D(u):
    """ Symmetrize with respect to (xyz) cycle. """
    try:
        nrm = np.linalg.norm(u.vector())
        V = u.function_space()
        assert V.mesh().topology().dim() == 3
        mesh1 = Mesh(V.mesh())
        mesh1.coordinates()[:, :] = mesh1.coordinates()[:, [1, 2, 0]]
        W1 = FunctionSpace(mesh1, 'CG', 1)

        # testing if symmetric
        bc = DirichletBC(V, 1, DomainBoundary())
        test = Function(V)
        bc.apply(test.vector())
        test = interpolate(Function(W1, test.vector()), V)
        assert max(test.vector()) - min(test.vector()) < 1.1

        v1 = interpolate(Function(W1, u.vector()), V)

        mesh2 = Mesh(mesh1)
        mesh2.coordinates()[:, :] = mesh2.coordinates()[:, [1, 2, 0]]
        W2 = FunctionSpace(mesh2, 'CG', 1)
        v2 = interpolate(Function(W2, u.vector()), V)
        pr = project(u+v1+v2)
        assert np.linalg.norm(pr.vector())/nrm > 0.01
        return pr
    except:
        print "Cyclic symmetrization failed!"
        return u


def rotational(u, n):
    """ Symmetrize with respect to n-fold symmetry. """
    # TODO: test one rotation only
    V = u.function_space()
    if V.mesh().topology().dim() > 2 or n < 2:
        return u
    mesh = V.mesh()
    sum = u
    nrm = np.linalg.norm(u.vector())
    rotation = np.array([[np.cos(2*np.pi/n), np.sin(2*np.pi/n)],
                         [-np.sin(2*np.pi/n), np.cos(2*np.pi/n)]])
    for i in range(1, n):
        mesh = Mesh(mesh)
        mesh.coordinates()[:, :] = np.dot(mesh.coordinates(), rotation)
        W = FunctionSpace(mesh, 'CG', 1)
        v = interpolate(Function(W, u.vector()), V)
        sum += v
    pr = project(sum)
    if np.linalg.norm(pr.vector())/nrm > 0.01:
        return pr
    else:
        return u
