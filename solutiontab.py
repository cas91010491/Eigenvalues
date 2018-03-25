""" Implementation of tab widget with solution data. """

try:
    from PyQt5.QtWidgets import QMainWindow, QSizePolicy, QVBoxLayout, QWidget
    from PyQt5.QtCore import pyqtSlot, QEvent, Qt
    from solTab5 import Ui_SolutionTab
except:
    from PyQt4.QtGui import QMainWindow, QSizePolicy, QVBoxLayout, QWidget
    from PyQt4.QtCore import pyqtSlot, QEvent, Qt
    from solTab import Ui_SolutionTab

from qvtk import QVTKRenderWindowInteractor
from tools import tooltips, addContextMenu, getWindow
from solver import shiftMesh, symmetrize
from viper3d import Viper3D
from paramchecker import evaluate

import numpy as np


class SolutionTab(QWidget, Ui_SolutionTab):

    """ Tab for solution data. """

    def __init__(self, dim):
        """ Build interface. """
        QWidget.__init__(self)
        self.setupUi(self)
        self.dim = dim
        if dim == 2:
            self.symZ.setParent(None)
            self.labelZ.setParent(None)
            self.symXZ.setParent(None)
            self.labelXZ.setParent(None)
            self.symYZ.setParent(None)
            self.labelYZ.setParent(None)
            self.symXYZ.setParent(None)
            self.labelXYZ.setParent(None)
            self.active = [self.symX, self.symY, self.symXY, self.symXXXX]
        else:
            self.rotations.setParent(None)
            self.symXXXX.setParent(None)
            self.labelRotations.setParent(None)
            self.active = [self.symX, self.symY, self.symZ, self.symXY,
                           self.symXZ, self.symYZ, self.symXYZ]

        self.eigList.installEventFilter(self)
        self.numContours.installEventFilter(self)

        # VTK widget
        self.eigenWidget = QVTKRenderWindowInteractor(self.eigenFrame)
        self.eigenWidget.setToolTip(tooltips['vtk'])
        addContextMenu(self.eigenWidget)

        vl = QVBoxLayout()
        vl.setContentsMargins(0, 0, 0, 0)
        vl.addWidget(self.eigenWidget)
        self.eigenFrame.setLayout(vl)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.eigenWidget.setSizePolicy(sizePolicy)
        self.eigenWidget.setHidden(True)
        self.eigenWidget.installEventFilter(getWindow())
        self.eigenWidget.viper = None

        self.lastItem = -1
        self.lastSym = str(self.getSymmetrizations())
        self.data = ""
        self.lastFunctional = "1"

        self.contours.setToolTip(tooltips['contours'])
        self.numContours.setToolTip(tooltips['contours'])
        self.functional.setToolTip(tooltips['eigenvalueFunctional'])

    def formatData(self):
        """ Fill domain details with information from self.data. """
        domain = self.data['domain']
        size = self.data['size']
        trans = self.data['transforms']
        self.domainData.setText(domain[0] + ' with ' + size[6] + ' cells')
        if len(trans):
            self.domainData.append('<strong>Transformations used!</strong>')
        self.domainData.append('Parameters: ' + domain[1])

        self.domainData.append('<br><strong>Geometric properties:</strong>')
        geometry = self.data['geometry']
        if self.dim == 2:
            self.domainData.append('Area: ' + str(geometry['A']))
            self.domainData.append('Perimeter: ' + str(geometry['P']))
        else:
            self.domainData.append('Volume: ' + str(geometry['A']))
            self.domainData.append('Surface area: ' + str(geometry['P']))
        self.domainData.append('Center of mass: ' +
                               str(np.round(geometry['c'], 5)))
        self.domainData.append('Moment of intertia: ' +
                               str(geometry['I']))

        self.domainData.append('<br><strong>Initial mesh:</strong>')
        self.domainData.append(size[1] + ' refine to at least ' + size[0])
        self.domainData.append('size: ' + size[5] + ' cells')
        self.domainData.append('<strong>Final mesh:</strong>')
        self.domainData.append(size[3] + ' refine to ' + size[4]
                               + ' ' + size[2])
        self.domainData.append('size: ' + size[6] + ' cells')
        if size[7]:
            self.domainData.append(
                'Uniform refine before plotting.')

        self.domainData.append('<br><strong>Transformations:</strong>')
        if len(trans):
            for t in trans:
                self.domainData.append(t)
        else:
            self.domainData.append('None')

        self.domainData.append('<br><strong>Boundary conditions:</strong>')
        bcs = self.data['bcs']
        if len(bcs[0]):
            for b in bcs[0]:
                self.domainData.append(b)
            if len(trans):
                if bcs[1]:
                    self.domainData.append(
                        'Applied after transformations.')
                else:
                    self.domainData.append(
                        'Applied before transformations.')
        else:
            self.domainData.append('None')

        self.domainData.append('<br><strong>Solver:</strong>')

        self.domainData.append('<br><strong>Parsed domain parameters:</strong>')

    def eventFilter(self, source, event):
        """ Add/remove a few keyboard and mouse events. """
        if event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Return:
                # RETURN triggers Preview for eigenvalues
                if source in [self.eigList, self.numContours]:
                    self.on_genPlot_clicked()
                    return True
        return QMainWindow.eventFilter(self, source, event)

    def on_eigList_itemDoubleClicked(self, item):
        """ Double-click on eigenvalue plots the eigenfunction. """
        self.on_genPlot_clicked()

    def on_grayscale_stateChanged(self, state):
        """ Quickly redraw in color/bw. """
        self.on_genPlot_clicked()

    def on_flip_stateChanged(self, state):
        """ Quickly redraw with flipped colors. """
        self.on_genPlot_clicked()

    def on_contours_stateChanged(self, state):
        """ Quickly redraw in color/bw. """
        self.on_genPlot_clicked()

    def getSymmetrizations(self):
        """ Get order of symmetrizations from interface. """
        lst = [[], [], [], [], [], [], [], [], [], []]
        for s in self.active:
            if s.value() != 0:
                lst[abs(s.value())].append([[ord(c)-88
                                             for c in str(s.objectName())[3:]],
                                            s.value() > 0])
        lst = [s for l in lst for s in l]
        return lst

    @pyqtSlot()
    def on_genPlot_clicked(self):
        """ Generate the plot of the eigenfunction. """
        if self.eigList.currentItem() < 0:
            print "Nothing to plot! Select an eigenvalue first..."
            return
        lst = self.getSymmetrizations()
        if self.lastItem != self.eigList.currentItem() or \
                self.lastSym != str(lst):
            # recompute values at mesh vertices
            eigf = self.eigList.currentItem().eigenfunction
            mesh = eigf.function_space().mesh()
            dim = mesh.topology().dim()
            if len(lst):
                vec = np.array(self.data['geometry']['c'])
                shiftMesh(mesh, -vec)
                for d, sym in lst:
                    if len(d) > 3:
                        d = [0, 0, 0, self.rotations.value()]
                    eigf = symmetrize(eigf, d, sym)
                shiftMesh(mesh, vec)
            self.lastItem = self.eigList.currentItem()
            self.lastSym = str(lst)
            values = eigf.compute_vertex_values()

            self.eigenWidget.setHidden(False)
            hideAxes = False
            view = None
            v = self.eigenWidget.viper
            if v:
                hideAxes = v.hideAxes
                if v.dim == 3 or v.force3D:
                    view = v.getView()
                v.cleanUp()
                self.eigenWidget.viper = None
            v = Viper3D(mesh, dim, values, self.eigenWidget)
            v.hideAxes = hideAxes
            v.view = view
        else:
            # reuse old viper data
            v = Viper3D.copy(self.eigenWidget.viper)
            self.eigenWidget.viper.cleanUp()
            self.eigenWidget.viper = None

        v.useColor(not self.grayscale.isChecked(), self.flip.isChecked())
        v.setContours(self.numContours.value())
        if self.contours.isChecked():
            v.ContourPlot()
        else:
            v.SurfPlot()

        self.eigenWidget.viper = v

    @pyqtSlot(int)
    def on_symXXXX_valueChanged(self, value):
        """ Enable/disable rotations spin box. """
        self.rotations.setEnabled(value > 0)

    @pyqtSlot()
    def on_functional_returnPressed(self):
        """ Modify list of eigenvalues according to supplied functional. """
        f = self.functional.text()
        if not len(f):
            self.functional.setText(self.lastFunctional)
            return
        saved = f
        short = f[0] != ':'
        if short:
            f = 'e*' + f
        else:
            f = f[1:]
        e = np.array([self.eigList.item(i).eigenvalue
                      for i in range(self.eigList.count())])
        s = np.cumsum(e)
        subs = dict(self.data['geometry'])
        subs['V'] = subs['A']
        subs['L'] = subs['P']
        subs['S'] = subs['P']
        subs['e'] = e
        subs['s'] = s
        try:
            lst = evaluate(f, subs)
            assert len(lst) and (len(lst) == len(e) or short)
            e = np.copy(e)
            e[:] = np.NaN
            e[:len(lst)] = lst
            for i in range(self.eigList.count()):
                if abs(e[i]) < 1E-9:
                    e[i] = 0.0
                self.eigList.item(i).setText(str(i+1)+': '+str(e[i]))
            self.lastFunctional = saved
        except:
            pass
