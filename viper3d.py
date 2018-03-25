"""Custom plotting class based on original viper class from FEniCS."""
import vtk
import vtk.util.numpy_support as VN
from numpy import array, zeros


class Viper3D(object):

    """Main plotter class."""

    def cleanUp(self):
        """ It seems that VTK objects are not removed automatically. """
        self.iren.RemoveObservers("KeyPressEvent")
        self.ren.RemoveAllViewProps()
        self.ren = None
        del self.x
        del self.mesh
        import gc
        gc.collect()

    @classmethod
    def copy(cls, v, widget=None):
        """ Create a fresh copy of viper with the current camera view. """
        if widget is None:
            widget = v.widget
        newv = cls(v.mesh, v.datadim, v.x, widget, v.color)
        if v.dim == 3 or v.force3D:
            newv.view = v.getView()
        newv.last = v.last
        newv.hideAxes = v.hideAxes
        return newv

    def __init__(self, mesh, dim, data, widget, color=True):
        """Initialize plotter widget."""
        # for vtk arrays
        self.refs = []

        self.mesh = mesh
        self.dim = mesh.topology().dim()
        self.datadim = dim
        self.numContours = 10
        self.view = None
        self.force3D = False
        self.last = "self.ContourPlot()"
        self.flip = False
        self.hideAxes = False

        # grids for all dimensions
        self.makeGrids()

        # values for each grid
        self.x = data
        self.vtkgrid[0].GetPointData().SetScalars(VN.numpy_to_vtk(0*data))
        if dim == self.dim:
            self.vtkgrid[dim].GetPointData().SetScalars(VN.numpy_to_vtk(data))
        else:
            self.vtkgrid[dim].GetCellData().SetScalars(VN.numpy_to_vtk(data))
            self.refs.append(zeros((self.mesh.size(0), self.dim)))
            self.vtkgrid[self.dim].GetPointData().SetScalars(
                VN.numpy_to_vtk(self.refs[-1]))
            if self.dim == 3:
                self.refs.append(zeros((self.mesh.size(self.dim - dim),
                                        self.dim)))
                self.vtkgrid[self.dim - dim].GetCellData().SetScalars(
                    VN.numpy_to_vtk(self.refs[-1]))
            # maybe in others too

        # bounds for values
        if color:
            self.vmax = max(abs(self.x.min()), abs(self.x.max()))
            self.vmin = -self.vmax
        else:
            self.vmax = self.x.max()
            self.vmin = self.x.min()
        self.vrange = [self.x.min(), self.x.max()]
        # if 'vmin' in kwargs:
        #    self.vmin = kwargs['vmin']
        # if 'vmax' in kwargs:
        #     self.vmin = kwargs['vmax']

        # colors
        self.useColor(color)

        # colors for meshfunctions
        self.blue = vtk.vtkLookupTable()
        self.blue.SetNumberOfColors(2)
        self.blue.Build()
        self.blue.SetTableValue(0, 0.0, 0.0, 1.0, 0.2)
        self.blue.SetTableValue(1, 0.0, 0.0, 1.0, 1.0)

        self.blue3D = vtk.vtkLookupTable()
        self.blue3D.SetNumberOfColors(2)
        self.blue3D.Build()
        self.blue3D.SetTableValue(0, 0.0, 0.0, 1.0, 0.2)
        self.blue3D.SetTableValue(1, 0.4, 0.4, 1.0, 1.0)

        # colors for the boundary conditions
        self.bc = vtk.vtkLookupTable()
        self.bc.SetNumberOfColors(4)
        self.bc.Build()
        self.bc.SetTableValue(0, 0, 1, 0, 1.0)  # Steklov (values <0)
        self.bc.SetTableValue(1, 0, 0, 1, 0.2)  # Neumann/nothing
        self.bc.SetTableValue(2, 1, 0, 0, 1.0)  # Dirichlet
        self.bc.SetTableValue(3, 0, 0, 0, 1.0)  # Robin (values >=2)

        # scalarbar
        self.scalarbar = self.makeScalarbar(self.lut)
        # self.scalarbar.VisibilityOn()

        self.widget = widget

    def useColor(self, color, flip=False):
        """ Modify LUT for eigenfunction plots. """
        self.color = color
        self.flip = flip
        self.lut = vtk.vtkLookupTable()
        if color:
            self.vmax = max(abs(self.x.min()), abs(self.x.max()))
            self.vmin = -self.vmax
            vals = [x.split()
                    for x in open('res/gauss_120.lut', 'r').readlines()[1:]]
            self.lut.SetNumberOfColors(len(vals))
            self.lut.Build()
            for i in range(len(vals)):
                if len(vals[i]) == 4:
                    self.lut.SetTableValue(len(vals)-i-1 if flip else i,
                                           *[float(x) for x in vals[i]])
        else:
            self.vmax = self.x.max()
            self.vmin = self.x.min()
            self.lut.SetNumberOfColors(1025)
            self.lut.Build()
            for i in range(1025):
                val = i/1024.0
                self.lut.SetTableValue(1024-i if flip else i,
                                       val, val, val, 1.0)
            self.lut.SetTableRange(-512, 512)

    def makeScalarbar(self, lut):
        """Create color bar."""
        scalarbar = vtk.vtkScalarBarActor()
        scalarbar.SetLookupTable(lut)
        scalarbar.GetPositionCoordinate(
        ).SetCoordinateSystemToNormalizedViewport()
        scalarbar.GetPositionCoordinate().SetValue(0.1, 0.01)
        scalarbar.SetOrientationToHorizontal()
        scalarbar.SetWidth(0.8)
        scalarbar.SetHeight(0.14)
        scalarbar.VisibilityOff()
        scalarbar.GetTitleTextProperty().SetColor(0, 0, 0)
        scalarbar.GetLabelTextProperty().SetColor(0, 0, 0)
        return scalarbar

    # redo
    def simple_axis(self, ren):
        """Show axes."""
        if self.mesh is not None:
            tprop = vtk.vtkTextProperty()
            tprop.SetColor(0, 0, 0)
            tprop.ShadowOff()
            outline = vtk.vtkOutlineFilter()
            outline.SetInputData(self.vtkgrid[self.dim])
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputConnection(outline.GetOutputPort())
            self.axes = vtk.vtkCubeAxesActor2D()
            self.axes.SetInputConnection(normals.GetOutputPort())
            self.axes.SetCamera(ren.GetActiveCamera())
            self.axes.GetProperty().SetColor(0, 0, 0)
            self.axes.SetAxisTitleTextProperty(tprop)
            self.axes.SetAxisLabelTextProperty(tprop)
            self.axes.SetCornerOffset(0)
            if self.dim == 2:
                self.axes.ZAxisVisibilityOff()
            ren.AddViewProp(self.axes)

    def makeGrids(self):
        """Build grids based on mesh."""
        # make sure connectivity was created
        self.mesh.init()
        # vertices
        cl = zeros((self.mesh.size(0), 3), dtype='d')
        cl[:, :self.dim] = self.mesh.coordinates()
        # keep reference
        self.refs.append(cl)
        # make vtkarray
        v = vtk.vtkPoints()
        v.SetNumberOfPoints(len(cl))
        v.SetData(VN.numpy_to_vtk(cl))
        # add points to a new grid
        self.vtkgrid = [None] * (self.dim + 1)
        # grids for edges, faces, cells
        for dim in range(1, self.dim + 1):
            self.vtkgrid[dim] = vtk.vtkUnstructuredGrid()
            # grids share points
            self.vtkgrid[dim].SetPoints(v)
            # get connectivity from topology
            nl = array(self.mesh.topology()(dim, 0)()).reshape(-1, dim + 1)
            ncells = len(nl)
            # cellsize = dim + 2
            cells = zeros((ncells, dim + 2), dtype=VN.ID_TYPE_CODE)
            cells[:, 1:] = nl
            cells[:, 0] = dim + 1
            self.refs.append(cells)
            # vtk cell array
            ca = vtk.vtkCellArray()
            ca.SetCells(ncells, VN.numpy_to_vtkIdTypeArray(cells))
            # add edges/faces as VTK cells
            if dim == 1:
                self.vtkgrid[dim].SetCells(vtk.VTK_LINE, ca)
            elif dim == 2:
                self.vtkgrid[dim].SetCells(vtk.VTK_TRIANGLE, ca)
            else:
                self.vtkgrid[dim].SetCells(vtk.VTK_TETRA, ca)
        self.vtkgrid[0] = self.vtkgrid[self.dim]

    def key_press_methods(self, obj, event):
        """Handle keypresses for plots."""
        key = obj.GetKeyCode()
        if key in ['a', 'A']:
            try:
                self.axes.SetVisibility(not self.axes.GetVisibility())
                self.hideAxes = not self.hideAxes
                self.renWin.Render()
            except:
                pass
        elif key in ['r', 'R']:
            self.setView(self.initview)
            self.ren.ResetCamera()
            self.renWin.Render()
        elif key in ['i', 'I']:
            # change interaction type (rubberband?)
            pass
        elif key in ['p', 'P'] and (self.dim == 3 or self.force3D):
            self.ren.GetActiveCamera().SetParallelProjection(
                not self.ren.GetActiveCamera().GetParallelProjection())
            self.renWin.Render()

    def Contours(self, num, opacity=0.2):
        """Create contours."""
        contour = vtk.vtkMarchingContourFilter()
        contour.SetInputData(self.vtkgrid[self.dim])
        r = (self.vrange[1]-self.vrange[0]) / 2.0 / num
        if num == 1:
            contour.SetValue(0, 0)
        else:
            contour.GenerateValues(
                num, (self.vrange[0] + r, self.vrange[1] - r))
        contour.ComputeScalarsOn()
        contour.UseScalarTreeOn()
        contour.Update()
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(contour.GetOutputPort())
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())
        mapper.SetLookupTable(self.lut)
        # bw contours are barely visible without this modification
        # also account for flip
        if not self.color:
            if self.flip:
                mapper.SetScalarRange(2*self.vmin-self.vmax, self.vmax)
            else:
                mapper.SetScalarRange(self.vmin, 2*self.vmax-self.vmin)
        else:
            mapper.SetScalarRange(self.vmin, self.vmax)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetLineWidth(3)
        return actor

    # good for mesh plot and boundary conditions
    def MeshFunction(self, dim, opacity=1, lut=None, min=0, max=1):
        """Create colored mesh."""
        domain = vtk.vtkGeometryFilter()
        domain.SetInputData(self.vtkgrid[dim])
        # domain.MergingOff()
        domain.Update()
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(domain.GetOutputPort())
        # mapper for domain
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())
        # use cell values for edge and face functions, but not cell functions
        if dim < self.dim:
            mapper.SetScalarModeToUseCellData()
        if lut is None:
            lut = self.blue
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(min, max)
        # actor for domain
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        if not self.color:
            opacity = 1-(1-opacity)/2.0
        actor.GetProperty().SetOpacity(opacity)
        return actor

    def Domain(self, dim, opacity=0.2, lut=None, warp=False):
        """Create domain or faces."""
        domain = vtk.vtkGeometryFilter()
        domain.SetInputData(self.vtkgrid[dim])
        domain.Update()
        mapper = vtk.vtkPolyDataMapper()
        if warp:
            warp = vtk.vtkWarpScalar()
            warp.SetInputConnection(domain.GetOutputPort())
            warp.SetScaleFactor(1/2.0/self.vmax)
            mapper.SetInputConnection(warp.GetOutputPort())
        else:
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputConnection(domain.GetOutputPort())
            mapper.SetInputConnection(normals.GetOutputPort())

        if lut is None:
            lut = self.lut
        if self.dim == 3:
            mapper.SetLookupTable(self.lut)
            mapper.SetScalarRange(self.vmin, self.vmax)
        else:
            mapper.SetLookupTable(self.lut)
            mapper.SetScalarRange(self.vmin, self.vmax)
        if dim == 0:
            mapper.SetLookupTable(self.blue)
            mapper.SetScalarRange(0.0, 1.0)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        if not self.color and self.dim == 2:
            opacity = 1-(1-opacity)/2.0
        actor.GetProperty().SetOpacity(opacity)
        return actor

    # edges of polyhedrons
    def Edges(self, opacity=1):
        """Create edges of polytopes."""
        domain = vtk.vtkGeometryFilter()
        domain.SetInputData(self.vtkgrid[self.dim])
        domain.Update()
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(domain.GetOutputPort())
        edges = vtk.vtkFeatureEdges()
        edges.SetInputConnection(normals.GetOutputPort())
        edges.ManifoldEdgesOff()
        edges.BoundaryEdgesOn()
        edges.NonManifoldEdgesOff()
        edges.FeatureEdgesOff()
        edges.SetFeatureAngle(1)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(edges.GetOutputPort())
        # if self.mesh.topology().dim()==3:
        mapper.ScalarVisibilityOff()
        # else:
        # mapper.SetLookupTable(self.lut)
        # mapper.SetScalarRange(self.vmin, self.vmax)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(opacity)
        return actor

    def Render(self, actors, force3D=False):
        """Render the plot."""
        self.ren = vtk.vtkRenderer()
        self.force3D = force3D

        self.ren.SetBackground(1, 1, 1)
        for a in actors:
            self.ren.AddActor(a)
        # self.ren.AddActor2D(self.scalarbar)
        self.renWin = self.widget.GetRenderWindow()
        self.renWin.AddRenderer(self.ren)

        # depth peeling
        self.renWin.SetAlphaBitPlanes(True)
        self.renWin.SetMultiSamples(0)
        self.ren.UseDepthPeelingOn()
        self.ren.SetMaximumNumberOfPeels(10)
        self.ren.SetOcclusionRatio(0.1)

        # set camera
        # z-axis pointing up in 3D
        if self.dim == 3:
            self.ren.GetActiveCamera().SetViewUp((0.0, 0.0, 1.0))
            self.ren.GetActiveCamera().SetPosition((100.0, 50.0, 50.0))
        else:
            self.ren.GetActiveCamera().SetParallelProjection(True)
        self.ren.ResetCamera()
        self.initview = self.getView()

        # use saved camera view if plot is 3D
        if self.view is not None and (force3D or self.dim == 3):
            self.setView(self.view)

        # self.renWin.SetWindowName(self.title)
        self.iren = self.renWin.GetInteractor()
        if self.mesh.topology().dim() == 3 or force3D:
            style = vtk.vtkInteractorStyleTrackballCamera()
        else:
            style = vtk.vtkInteractorStyleImage()
        # style = vtk.vtkInteractorStyleUnicam()
        self.iren.SetInteractorStyle(style)
        self.iren.RemoveObservers("CharEvent")
        self.iren.RemoveObservers("KeyPressEvent")
        self.iren.AddObserver("KeyPressEvent", self.key_press_methods)
        self.iren.Initialize()
        self.simple_axis(self.ren)
        if self.hideAxes:
            self.axes.SetVisibility(not self.axes.GetVisibility())

        self.renWin.Render()

    def getView(self):
        """ Return the current camera viewing data. """
        camera = self.ren.GetActiveCamera()
        return [camera.GetViewUp(), camera.GetPosition(),
                camera.GetFocalPoint(),  camera.GetViewAngle(),
                camera.GetClippingRange(), camera.GetParallelScale(),
                camera.GetParallelProjection()]

    def setView(self, data):
        """ Return the current camera viewing data. """
        camera = self.ren.GetActiveCamera()
        camera.SetViewUp(data[0])
        camera.SetPosition(data[1])
        camera.SetFocalPoint(data[2])
        camera.SetViewAngle(data[3]),
        camera.SetClippingRange(data[4])
        camera.SetParallelScale(data[5]),
        camera.SetParallelProjection(data[6])

    def write_png(self, filename, magnify=2):
        """Save plot as png."""
        # FIXME: retina causes artefacts near edges
        large = vtk.vtkRenderLargeImage()
        large.SetInputData(self.ren)
        large.SetMagnification(magnify)
        png = vtk.vtkPNGWriter()
        png.SetFileName(filename)
        png.SetInputConnection(large.GetOutputPort())
        png.Write()

    def setContours(self, num):
        """ Set number of contours for ContourPlot. """
        self.numContours = num

    def ContourPlot(self, num=None, mesh=False, edges=True):
        """Plot contours and the domain."""
        if num is not None:
            self.numContours = num
        if self.mesh.topology().dim() == 3:
            opacity = 0.2
            domainOpacity = 0.15
        else:
            opacity = 1
            domainOpacity = 0.25
        actor = self.Contours(self.numContours, opacity)
        actors = [actor, self.Domain(self.dim, domainOpacity)]
        if self.mesh.topology().dim() == 3 and edges:
            actors.append(self.Edges())
        self.Render(actors)
        self.last = "self.ContourPlot()"

    def SurfPlot(self):
        """Plot warped surface or surface of the solid. """
        if self.dim == 2:
            actor = self.Domain(2, opacity=1, warp=True)
        else:
            actor = self.Domain(3, opacity=1, warp=False)
        self.Render([actor], force3D=True)
        self.last = "self.SurfPlot()"

    def MeshPlot(self):
        """Plot a colored mesh."""
        actors = []
        if self.dim == 2:
            actors.append(self.Domain(0, opacity=0.8))
            actors.append(self.MeshFunction(self.datadim))
        else:
            actors.append(self.MeshFunction(self.datadim, opacity=1, lut=self.blue3D))
            actors.append(self.Domain(1))
            actors.append(self.Edges())
        self.Render(actors)
        self.last = "self.MeshPlot()"

    def BCPlot(self):
        """Plot a colored mesh."""
        actors = []
        if self.dim == 2:
            actors.append(self.Domain(0, opacity=0.4))
            actors.append(self.MeshFunction(self.datadim, lut=self.bc,
                          min=-1, max=2))
        else:
            actors.append(self.Domain(1))
            actors.append(self.MeshFunction(self.datadim, opacity=1,
                                            lut=self.bc, min=-1, max=2))
            actors.append(self.Edges())
        self.Render(actors)
        self.last = "self.BCPlot()"

    def LastPlot(self):
        """ Use previous plot type. """
        eval(self.last)
