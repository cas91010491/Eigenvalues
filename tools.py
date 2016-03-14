""" Helper functions for the main application. """
from inspect import cleandoc
import re

try:
    from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, \
        QVBoxLayout, QSizePolicy, QAction
    from PyQt5.QtCore import Qt
except:
    from PyQt4.QtGui import QApplication, QDialog, QFileDialog, \
        QVBoxLayout, QSizePolicy, QAction
    from PyQt4.QtCore import Qt

from qvtk import QVTKRenderWindowInteractor
from viper3d import Viper3D


def getWindow():
    """ Return main application window. """
    for widget in QApplication.topLevelWidgets():
        if widget.objectName() == 'MainWindow':
            return widget


def newWindow(parent):
    """ Open a plot in a new window. """
    print QApplication.topLevelWidgets()
    plotWindow = QDialog(getWindow())
    plotWidget = QVTKRenderWindowInteractor(plotWindow)
    plotWidget.setToolTip(tooltips['vtk'])
    addContextMenu(plotWidget)
    plotWidget.installEventFilter(getWindow())

    vl = QVBoxLayout()
    vl.setContentsMargins(0, 0, 0, 0)
    vl.addWidget(plotWidget)
    plotWindow.setLayout(vl)
    sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    plotWidget.setSizePolicy(sizePolicy)
    plotWindow.show()
    plotWindow.raise_()
    v = Viper3D.copy(parent.viper, widget=plotWidget)
    plotWidget.viper = v
    v.LastPlot()


def addContextMenu(widget):
    """ Add context menu items to QVTK plot widgets. """
    widget.setContextMenuPolicy(Qt.ActionsContextMenu)
    saveAction = QAction("Save as ...", widget)
    saveAction.triggered.connect(lambda x: savePlot(widget))
    widget.addAction(saveAction)
    newWinAction = QAction("Open in new window", widget)
    newWinAction.triggered.connect(
        lambda x: newWindow(widget))
    widget.addAction(newWinAction)


def openDialog(name, filter):
    """ Create non-native open file dialog. """
    qfd = QFileDialog(getWindow(), name)
    qfd.setNameFilter(filter)
    qfd.setFileMode(QFileDialog.ExistingFile)
    qfd.setOptions(QFileDialog.DontUseNativeDialog)
    qfd.setAcceptMode(QFileDialog.AcceptOpen)
    qfd.setViewMode(QFileDialog.List)
    return qfd


def saveDialog(name, filter):
    """ Create non-native save file dialog. """
    qfd = QFileDialog(getWindow(), name)
    qfd.setNameFilter(filter)
    qfd.setFileMode(QFileDialog.AnyFile)
    qfd.setOptions(QFileDialog.DontUseNativeDialog)
    qfd.setAcceptMode(QFileDialog.AcceptSave)
    qfd.setViewMode(QFileDialog.List)
    return qfd


def savePlot(parent):
    """ Save a previewed mesh to a file. """
    # save = saveDialog("Save as ...", ".png;;.pdf")
    save = saveDialog("Save as ...", ".png")
    if save.exec_() != QDialog.Accepted:
        return
    name = str(save.selectedFiles()[0])
    if not re.search(r'\.[a-zA-Z0-9]{1,3}$', name):
        name += save.selectedNameFilter()
    if name[-3:] == 'png':
        parent.viper.write_png(name)


tooltips = {
    'vtk': """
    Right-click: Context menu
    Left-click with Shift and/or Command: Shift/Rotate/Zoom
    R: Reset camera
    P: Toggle parallel/perspective projection
    A: Toggle axes visibility (if available)
    """,
    'contours': """
    Setting number of countours to 1 ensures nodal set will be shown.
    """,
    'copyParam': """
    Copy parameters from this solution to the main parameters tabs.
    """,
    'eigenvalueFunctional': r"""
    Rescale eigenvalues using one of the geometric factors:
    A/V: area/volume
    P/L/S: perimeter, surface area
    I: moment of inertia with respect to the center of mass
    G0/G1/Grobin/Gsteklov: scaling factor from papers by Laugesen and Siudeja

    If the field starts with ':' then an eigenvalue functional can be specified:
    e: eigenvalues
    s: sum of eigenvalues
    e[:-1]: all but last eigenvalue
    e[1:]: all but first eigenvalue
    e[i]: i-th eigenvalue
    s[i]: i-th sum
    The expression is interpreted as elementwise array computation supporting \
    Python slicing and any numpy function. Caution: indexing starts from 0.

    E.g.
    ':(e[1:]-e[:-1])*A' lists gaps between eigenvalues scaled using area,
    ':s' lists sums of eigenvalue.
    ':cumprod(e[1:])' lists products of eigenvalues, skipping the first one

    Important: Eigenfunctions do not change, even when the new list is shorter.
    """,
    }

for key in tooltips:
    tooltips[key] = re.sub(r'\\\n', '', cleandoc(tooltips[key]))
