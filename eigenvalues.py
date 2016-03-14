#!/usr/bin/python
"""
Main application file.

Should work with PyQt4 and PyQt5.
Requires FEniCS 1.6.
"""
from __future__ import division

import sys
import os

# PyQt4/5 setup
try:
    from PyQt5.QtWidgets import QMainWindow, QApplication, QSizePolicy, \
        QVBoxLayout, QListWidgetItem, QDialog
    from PyQt5.QtGui import QIntValidator, QDoubleValidator, QIcon
    from PyQt5.QtCore import pyqtSlot, QEvent, Qt, QSize, QPoint, \
        QCoreApplication
    from eigGUI5 import Ui_MainWindow
except:
    from PyQt4.QtGui import QMainWindow, QApplication, QSizePolicy, QIcon, \
        QVBoxLayout, QListWidgetItem, QDialog, QIntValidator, QDoubleValidator
    from PyQt4.QtCore import pyqtSlot, QEvent, Qt, QSize, QPoint, \
        QCoreApplication
    from eigGUI import Ui_MainWindow

from qvtk import QVTKRenderWindowInteractor

# QSettings for saving all fields
from config import QSettingsManager
QCoreApplication.setOrganizationName("Siudej")
QCoreApplication.setOrganizationDomain("siudej.com")
QCoreApplication.setApplicationName("Eigenvalues")
settings = QSettingsManager()

from viper3d import Viper3D
from domains import build_domain_dict, build_mesh
from transforms import build_transform_dict, transform_mesh
from solver import refine_mesh, Solver  # , USE_EIGEN
from boundary import build_bc_dict, bcApplyChoices, marked_boundary, \
    mark_conditions
from longcalc import LongCalculation, pickle_mesh, pickle_solutions
from tools import tooltips, openDialog, addContextMenu
from solutiontab import SolutionTab

from dolfin import set_log_level, parameters
parameters['allow_extrapolation'] = True
# solver will have an option to switch to EIGEN
# if USE_EIGEN:
#     parameters['linear_algebra_backend'] = 'Eigen'
set_log_level(30)

# fix for missing qt plugins in app
if not QApplication.libraryPaths():
    QApplication.addLibraryPath(str(os.environ['RESOURCEPATH'] + '/qt_plugins'))


class MainWindow(QMainWindow, Ui_MainWindow):

    """ Main application window. """

    def __init__(self):
        """ Initialize main window. """
        QMainWindow.__init__(self)
        self.mesh = None
        self.dim = 2
        self.solver = None

        # Set up the user interface from Designer.
        self.setupUi(self)
        self.resize(QSize(settings.get('citewidth'),
                          settings.get('citeheight')))
        self.move(QPoint(settings.get('citex'),
                         settings.get('citey')))
        self.setWindowTitle('Eigenvalues (powered by FEniCS)')
        path = os.path.dirname(os.path.realpath(__file__))+"/res/"
        self.setWindowIcon(QIcon(path + "eig.icns"))

        self.addVTK()

        self.populateLists()

        self.show()
        self.raise_()

        # event filters
        self.domains.installEventFilter(self)
        self.parameters.installEventFilter(self)
        self.meshWidget.installEventFilter(self)
        self.transformedMeshWidget.installEventFilter(self)
        self.bcWidget.installEventFilter(self)

        # find all widgets with data and add to settings
        for name, widget in self.__dict__.items():
            try:
                if widget.__class__.__name__ in ('QComboBox', 'QLineEdit',
                                                 'QSpinBox', 'QCheckBox'):
                    settings.add_handler(name, widget)
            except:
                pass

    def closeEvent(self, event):
        """ Make sure settings are saved. """
        settings.set('citex', self.pos().x())
        settings.set('citey', self.pos().y())
        settings.set('citeheight', self.size().height())
        settings.set('citewidth', self.size().width())
        event.accept()

    def addVTK(self):
        """ Add VTK widgets and their actions. """
        # domain plots
        self.meshWidget = QVTKRenderWindowInteractor(self.meshFrame)
        self.meshWidget.setToolTip(tooltips['vtk'])
        addContextMenu(self.meshWidget)

        vl = QVBoxLayout()
        vl.setContentsMargins(0, 0, 0, 0)
        self.meshFrame.setLayout(vl)
        self.meshFrame.layout().addWidget(self.meshWidget)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.meshWidget.setSizePolicy(sizePolicy)
        self.meshWidget.setHidden(True)
        self.meshWidget.viper = None

        # transformation plots
        self.transformedMeshWidget = QVTKRenderWindowInteractor(
            self.transformedMeshFrame)
        self.transformedMeshWidget.setToolTip(tooltips['vtk'])
        addContextMenu(self.transformedMeshWidget)

        vl = QVBoxLayout()
        vl.setContentsMargins(0, 0, 0, 0)
        vl.addWidget(self.transformedMeshWidget)
        self.transformedMeshFrame.setLayout(vl)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.transformedMeshWidget.setSizePolicy(sizePolicy)
        self.transformedMeshWidget.setHidden(True)
        self.transformedMeshWidget.viper = None

        # boundary plots
        self.bcWidget = QVTKRenderWindowInteractor(self.bcFrame)
        self.meshWidget.setToolTip(tooltips['vtk'])
        addContextMenu(self.bcWidget)

        vl = QVBoxLayout()
        vl.setContentsMargins(0, 0, 0, 0)
        vl.addWidget(self.bcWidget)
        self.bcFrame.setLayout(vl)
        self.bcWidget.setSizePolicy(sizePolicy)
        self.bcWidget.setHidden(True)
        self.bcWidget.viper = None

    def populateLists(self):
        """ Add domains and transforms to widgets, initialize other fields. """
        # domains
        self.domains.setSortingEnabled(True)
        self.domains_dict = build_domain_dict()
        for key in self.domains_dict.keys():
            self.domains.addItem(key)
        # start with a preselected domain
        self.domains.setCurrentRow(0)
        self.tabs.tabBar().setCurrentIndex(0)
        self.domains.setFocus()
        self.description.setPlainText(
            self.domains_dict[str(self.domains.currentItem().text())].help)
        self.initSize.setValidator(QIntValidator(1, 100000))
        self.initSize.setText('1')
        self.meshSize.setValidator(QIntValidator(1, 1000000))
        self.meshSize.setText('1000')
        self.targetValue.setValidator(QDoubleValidator())

        # transforms
        self.transforms.setSortingEnabled(True)
        self.transforms_dict = build_transform_dict()
        for key in self.transforms_dict.keys():
            self.transforms.addItem(key)
        # preselected transform
        self.transforms.setCurrentRow(0)
        self.transformDescription.setPlainText(
            self.transforms_dict[
                str(self.transforms.currentItem().text())].help)

        # boundary conditions
        self.bcs.setSortingEnabled(True)
        self.bc_dict = build_bc_dict()
        for key in self.bc_dict.keys():
            self.bcs.addItem(key)
        self.bcApplyTo.addItems(bcApplyChoices)
        # preselected bc
        self.bcs.setCurrentRow(0)
        self.bcDescription.setPlainText(
            self.bc_dict[str(self.bcs.currentItem().text())].help)

    #
    #
    # domains interface
    #
    #

    def on_domains_currentItemChanged(self, current, previous):
        """ Domain change event. """
        self.description.setPlainText(
            self.domains_dict[str(current.text())].help)
        if self.domains_dict[str(current.text())].with_params():
            self.parameters.lineEdit().setText(
                self.domains_dict[str(current.text())].params)
            self.parameters.setEnabled(True)
        else:
            self.parameters.lineEdit().setText("")
            self.parameters.setEnabled(False)

    def on_domains_itemDoubleClicked(self, item):
        """ Double-click on domain starts preview. """
        self.on_preview_clicked()

    def on_initSize_returnPressed(self):
        """ Pressing RETURN on mesh size starts preview. """
        self.on_preview_clicked()

    def getMesh(self):
        """
        Get mesh from appropriate domain.

        Ask for a file in domain needs a file.
        """
        params = str(self.parameters.currentText())
        domain = self.domains_dict[str(self.domains.currentItem().text())]
        if self.initSize.text():
            size = int(self.initSize.text())
        else:
            size = 1
        if domain.dim == "FILE":
            open = openDialog(domain.dialog, domain.default)
            if open.exec_() == QDialog.Accepted:
                params = str(open.selectedFiles()[0])
            else:
                params = ""
            if not params:
                self.stats.appendPlainText("Mesh generation failed!\n\n")
                return 0
            self.parameters.lineEdit().setText(params)
        else:
            params = str(self.parameters.currentText())

        # process domain parameters
        domain.eval(params)
        # multiprocessing worker / progress dialog
        # calling domain object (building mesh) may be time consuming
        longcalc = LongCalculation(
            domain, [], pickle_mesh, "Generating mesh")
        code = longcalc.exec_()
        if code:
            cells, vertices = longcalc.res
            # rebuild the mesh after unpickling its cells and vertices
            self.mesh = build_mesh(cells, vertices)
        else:
            # worker failed
            longcalc.cleanUp()
            self.stats.appendPlainText("Mesh generation failed!\n\n")
            return 0
        if not params:
            self.parameters.lineEdit().setText(domain.params)
        else:
            if self.parameters.findText(domain.params) == -1:
                self.parameters.insertItem(0, domain.params)
        self.dim = self.mesh.topology().dim()
        # refine mesh
        edge = self.initRefine.currentIndex() > 0
        self.mesh = refine_mesh(self.mesh, size, edge)
        self.transMesh = None
        dim = self.mesh.topology().dim()
        self.stats.appendPlainText("Mesh generated with "
                                   + str(self.mesh.size(dim)) + " cells"
                                   "\nDomain: " + domain.name +
                                   "\nParsed: " + str(domain.values) +
                                   "\nFrom: " + domain.params + "\n\n")
        self.domain = domain
        return 1

    @pyqtSlot()
    def on_preview_clicked(self):
        """ Preview mesh in VTK widget. """
        self.meshWidget.setHidden(False)
        if not self.getMesh():
            return
        if self.meshWidget.viper:
            self.meshWidget.viper.iren.RemoveObservers("KeyPressEvent")
            self.meshWidget.viper.cleanUp()
            self.meshWidget.viper = None
        val = marked_boundary(self.mesh)
        v = Viper3D(self.mesh, self.dim - 1, val, self.meshWidget)
        v.MeshPlot()
        self.meshWidget.viper = v

    #
    #
    # transforms interface
    #
    #

    def on_transforms_itemDoubleClicked(self, item):
        """ Double-click on transform name adds it to selected transforms. """
        name = str(item.text())
        obj = self.transforms_dict[name]()
        new = QListWidgetItem(name + ": " + obj.default)
        new.obj = obj
        self.selectedTransforms.addItem(new)
        self.selectedTransforms.setCurrentRow(
            self.selectedTransforms.count()-1)
        # self.transformParameters.setText(obj.default)
        self.transformParameters.setFocus()

    def on_selectedTransforms_itemDoubleClicked(self, item):
        """ Double-click on selected transform removes it from list. """
        # item needs to be explicitely deleted
        item = self.selectedTransforms.takeItem(
            self.selectedTransforms.currentRow())
        del item

    def on_transforms_currentItemChanged(self, current, previous):
        """ Selecting a transform updates description. """
        self.transformDescription.setPlainText(
            self.transforms_dict[str(current.text())].help)

    def on_selectedTransforms_currentItemChanged(self, current, previous):
        """
        Update fields when current transform changes.

        Selecting one of the chosen transforms puts it's parameters in
        edit line, and it's help in description.
        """
        if current is None:
            return
        self.transformDescription.setPlainText(
            self.transforms_dict[str(current.obj.name)].help)
        self.transformParameters.setText(current.obj.params)

    def on_transformParameters_returnPressed(self):
        """ RETURN on transform parameters updates the selected transform. """
        self.on_updateTransformParams_clicked()

    @pyqtSlot()
    def on_updateTransformParams_clicked(self):
        """ Try to update selected transform with current parameters. """
        item = self.selectedTransforms.currentItem()
        if item is None:
            return
        item.obj.update(str(self.transformParameters.text()))
        item.setText(item.obj.name + ': ' + item.obj.params)

    def getTransMesh(self):
        """ Apply transformations to the domain. """
        if self.mesh is None:
            self.getMesh()
        lst = [self.selectedTransforms.item(i).obj for
               i in xrange(self.selectedTransforms.count())]
        self.transMesh = transform_mesh(self.mesh, lst)
        return 1

    @pyqtSlot()
    def on_previewTransformed_clicked(self):
        """ Preview transformed mesh in VTK widget. """
        self.getTransMesh()
        self.transformedMeshWidget.setHidden(False)
        val = marked_boundary(self.transMesh)
        if self.transformedMeshWidget.viper:
            self.transformedMeshWidget.viper.iren.RemoveObservers(
                "KeyPressEvent")
            self.transformedMeshWidget.viper.cleanUp()
            self.transformedMeshWidget.viper = None
        v = Viper3D(self.transMesh, self.dim - 1, val,
                    self.transformedMeshWidget)
        v.MeshPlot()
        self.transformedMeshWidget.viper = v

    #
    #
    # bc interface
    #
    #

    @staticmethod
    def bcName(obj):
        """ Format then ame for the added boundary condition. """
        txt = obj.name
        if obj.param != "None":
            txt += ': ' + obj.param
        txt += '; ' + obj.applyTo
        if obj.applyTo != "global":
            txt += ': ' + obj.conditions
        return txt

    def on_bcs_itemDoubleClicked(self, item):
        """Double-click on boundary condition adds it to selected transforms."""
        name = str(item.text())
        obj = self.bc_dict[name]()
        new = QListWidgetItem(self.bcName(obj))
        new.obj = obj
        self.selectedBC.addItem(new)
        self.selectedBC.setCurrentRow(self.selectedBC.count()-1)
        self.bcParameters.setFocus()
        self.bcParts.setDisabled(True)  # global is default

    def on_selectedBC_itemDoubleClicked(self, item):
        """ Double-click on selected boundary cond. removes it from list. """
        # item needs to be explicitely deleted
        item = self.selectedBC.takeItem(self.selectedBC.currentRow())
        del item
        if self.selectedBC.count() == 0:
            index = self.bcApplyTo.findText("inside")
            self.bcApplyTo.removeItem(index)

    def on_bcs_currentItemChanged(self, current, previous):
        """ Selecting a boundary condition updates description. """
        self.bcDescription.setPlainText(
            self.bc_dict[str(current.text())].help)

    def on_selectedBC_currentItemChanged(self, current, previous):
        """
        Update fields when current boundary conditon changes.

        Selecting one of the chosen BC puts it's parameters in
        edit line, and it's help in description.
        """
        if current is None:
            return
        self.bcDescription.setPlainText(current.obj.help)
        if current.obj.param == "None":
            self.bcParameters.setDisabled(True)
        else:
            self.bcParameters.setDisabled(False)
            self.bcParameters.setText(current.obj.param)
        index = self.bcApplyTo.findText("inside")
        if current.obj.name == "Dirichlet":
            if index < 0:
                self.bcApplyTo.addItem("inside")
        else:
            self.bcApplyTo.removeItem(index)
        index = self.bcApplyTo.findText(current.obj.applyTo)
        self.bcApplyTo.setCurrentIndex(index)
        self.bcParts.setText(current.obj.conditions)

    def on_bcApplyTo_currentIndexChanged(self, index):
        """ Disable parts edit box for global conditions. """
        if self.bcApplyTo.currentText() == 'global':
            self.bcParts.setDisabled(True)
        else:
            self.bcParts.setDisabled(False)

    def on_bcParameters_returnPressed(self):
        """ RETURN on BC parameters updates the selected BC. """
        self.on_updateBCParams_clicked()

    def on_bcParts_returnPressed(self):
        """ RETURN on BC parts updates the selected BC. """
        self.on_updateBCParams_clicked()

    @pyqtSlot()
    def on_updateBCParams_clicked(self):
        """ Try to update selected boundary cond. with current parameters. """
        item = self.selectedBC.currentItem()
        if item is None:
            return
        item.obj.update(str(self.bcApplyTo.currentText()),
                        str(self.bcParts.text()),
                        str(self.bcParameters.text()))

        item.setText(self.bcName(item.obj))

    def getBCList(self):
        """ Build a list with boundary conditions. """
        lst = [self.selectedBC.item(i).obj for
               i in xrange(self.selectedBC.count())]
        return lst

    @pyqtSlot()
    def on_previewBC_clicked(self):
        """ Preview boundary conditions in VTK widget. """
        if self.useTransformed.isChecked():
            if self.transMesh is None:
                self.getTransMesh()
            mesh = self.transMesh
        else:
            if self.mesh is None:
                self.getMesh()
            mesh = self.mesh
        self.bcWidget.setHidden(False)
        val = mark_conditions(mesh, self.getBCList())
        if self.bcWidget.viper:
            self.bcWidget.viper.iren.RemoveObservers("KeyPressEvent")
            self.bcWidget.viper.cleanUp()
            self.bcWidget.viper = None
        v = Viper3D(mesh, self.dim - 1, val, self.bcWidget)
        v.BCPlot()
        self.bcWidget.viper = v

    #
    #
    # other options
    #
    #

    @pyqtSlot(int)
    def on_solveType_currentIndexChanged(self, index):
        """ Enable/disable target value field. """
        if index == 1:
            self.targetValue.setEnabled(True)
        else:
            self.targetValue.setEnabled(False)

    @pyqtSlot(int)
    def on_femType_currentIndexChanged(self, index):
        """ Enable/disable FEM degree for nonconforming. """
        if index > 0:
            self.femDegree.setDisabled(True)
            self.femDegree.setValue(1)
        else:
            self.femDegree.setDisabled(False)

    #
    #
    # Solve
    #
    #

    @pyqtSlot()
    def on_solve_clicked(self):
        """ Solve in domain tab. """
        domain = self.domains_dict[str(self.domains.currentItem().text())]
        if not (self.mesh is not None and domain.dim == 'FILE'):
            if not self.getMesh() and self.mesh is None:
                return
        dim = self.mesh.topology().dim()
        initsize = self.mesh.size(dim)
        trans = [self.selectedTransforms.item(i).obj
                 for i in xrange(self.selectedTransforms.count())]
        bcs = self.getBCList()
        # create solver and adjust parameters
        wTop, wBottom = self.getWeights()
        solver = Solver(self.mesh, bcs, trans, deg=self.femDegree.value(),
                        bcLast=self.useTransformed.isChecked(),
                        method=str(self.femType.currentText()),
                        wTop=wTop, wBottom=wBottom)
        solver.refineTo(int(self.meshSize.text()),
                        self.meshLimit.currentText() == 'at most',
                        self.refine.currentText() == 'long edge')
        if self.solveType.currentIndex() == 0:
            solver.solveFor(self.solveNumber.value(), None, False)
        else:
            solver.solveFor(self.solveNumber.value(),
                            float(self.targetValue.text()), False)
        # get ready for pickling
        solver.removeMesh()
        longcalc = LongCalculation(solver, [], pickle_solutions, "Solving")
        code = longcalc.exec_()
        if not code:
            # worker failed
            longcalc.cleanUp()
            self.stats.appendPlainText("Solver failed!\n\n")
            return
        results = longcalc.res
        eigv, eigf = results[:2]
        for i in range(len(eigf)):
            u = solver.newFunction()
            u.vector()[:] = eigf[i]
            eigf[i] = u
        finalsize = solver.finalsize
        sol = SolutionTab(dim)
        sol.data = {'geometry': results[2]}
        self.fillTabData(sol.data, trans, bcs, str(initsize), str(finalsize),
                         solver.extraRefine)
        sol.formatData()
        domain = self.domains.currentItem().text()
        self.solutionTabs.addTab(sol, domain)
        for i, [e, u] in enumerate(zip(eigv, eigf)):
            if abs(e) < 1E-9:
                e = 0.0
            new = QListWidgetItem(str(i+1)+': '+str(e))
            new.eigenvalue = e
            new.eigenfunction = u
            sol.eigList.addItem(new)
        self.tabs.tabBar().setCurrentIndex(4)
        self.solutionTabs.tabBar().setCurrentIndex(self.solutionTabs.count()-1)
        sol.setFocus(True)
        self.stats.appendPlainText("Solutions found.\n\n")

    def getWeights(self):
        """ Get weights for the Rayleigh quotient from the interface. """
        wTop = str(self.weightTop.currentText()).lower()
        wBottom = str(self.weightBottom.currentText()).lower()
        dim = self.mesh.topology().dim()
        if "none" in wTop:
            wTop = '1'
        elif "gaussian" in wTop:
            if dim == 2:
                wTop = 'exp(-(x[0]*x[0]+x[1]*x[1])/2)/(2*pi)'
            else:
                wTop = 'exp(-(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])/2)/pow(2*pi, 1.5)'
        if 'none' in wBottom:
            wBottom = '1'
        elif "gaussian" in wBottom:
            if dim == 2:
                wBottom = 'exp(-(x[0]*x[0]+x[1]*x[1])/2)/(2*pi)'
            else:
                wBottom = ('exp(-(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])/2)'
                           '/pow(2*pi, 1.5)')
        return wTop, wBottom

    def fillTabData(self, dct, transforms, bcs, initsize, finalsize, extraRef):
        """ Collect data used by solver. """
        dct['domain'] = [self.domain.name, self.domain.params,
                         self.domain.values]
        dct['size'] = [self.initSize.text(), self.initRefine.currentText(),
                       self.meshSize.text(), self.refine.currentText(),
                       self.meshLimit.currentText(), initsize, finalsize,
                       extraRef]
        dct['solver'] = [self.femType.currentText(), self.femDegree.value(),
                         self.solveNumber.value(), self.solveType.currentText(),
                         self.targetValue.text()]
        dct['transforms'] = [str(t) for t in transforms]
        dct['bcs'] = [[str(b) for b in bcs], self.useTransformed.isChecked()]

    def on_solutionTabs_tabCloseRequested(self, index):
        """ Remove a solution. """
        widget = self.solutionTabs.widget(index)
        self.solutionTabs.removeTab(index)
        widget.setParent(None)
        widget.deleteLater()

    @pyqtSlot()
    def on_solve2_clicked(self):
        """ Solve in transforms tab. """
        self.on_solve_clicked()

    @pyqtSlot()
    def on_solve3_clicked(self):
        """ Solve in boundary tab. """
        self.on_solve_clicked()

    @pyqtSlot()
    def on_solve4_clicked(self):
        """ Solve in other tab. """
        self.on_solve_clicked()

    #
    #
    # various functions
    #
    #

    def eventFilter(self, source, event):
        """ Add/remove a few keyboard and mouse events. """
        if event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Return:
                # domains and parameters: RETURN triggers Preview
                if source in [self.domains, self.parameters]:
                    self.on_preview_clicked()
                    return True
        if event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.RightButton:
                # stop VTK from capturing right click
                if source.__class__.__name__ == 'QVTKRenderWindowInteractor':
                    return True
        return QMainWindow.eventFilter(self, source, event)


if __name__ == "__main__":
    # billiard instead of multiprocessing to avoid BLAS errors due to fork
    import billiard
    billiard.forking_enable(False)
    settings.set_default('citeheight', 600)
    settings.set_default('citewidth', 800)
    settings.set_default('citex', 0)
    settings.set_default('citey', 0)
    app = QApplication(sys.argv)
    path = os.path.dirname(os.path.realpath(__file__))+"/res/"
    app.setWindowIcon(QIcon(path + "eig.icns"))
    window = MainWindow()
    sys.exit(app.exec_())
