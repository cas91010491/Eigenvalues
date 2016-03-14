"""
A simple VTK widget for PyQt v4, the Qt v4 bindings for Python.
See http://www.trolltech.com for Qt documentation, and
http://www.riverbankcomputing.co.uk for PyQt.

This class is based on the vtkGenericRenderWindowInteractor and is
therefore fairly powerful.  It should also play nicely with the
vtk3DWidget code.

Created by Prabhu Ramachandran, May 2002
Based on David Gobbi's QVTKRenderWidget.py

Changes by Gerard Vermeulen Feb. 2003
 Win32 support.

Changes by Gerard Vermeulen, May 2003
 Bug fixes and better integration with the Qt framework.

Changes by Phil Thompson, Nov. 2006
 Ported to PyQt v4.
 Added support for wheel events.

Changes by Phil Thompson, Oct. 2007
 Bug fixes.

Changes by Phil Thompson, Mar. 2008
 Added cursor support.

Changes by Rodrigo Mologni, Sep. 2013 (Credit to Daniele Esposti)
 Bug fix to PySide: Converts PyCObject to void pointer.

Changes by Bartlomiej Siudeja, Jan. 2015
 Added PyQt5 handling.
 Added retina display detection (automatic in PyQt5) and retina keyword.
    In PyQt4 retina=True should be added to the constructor for retina displays.
    Is there a way to detect retina in PyQt4?
    Widget shows quarter-sized rendering without retina display handling.
    Implementation is a hack, and should be improved.
"""

try:
    from PyQt5.QtCore import Qt, QTimer, QSize, QEvent
    from PyQt5.QtWidgets import QWidget, QApplication, QSizePolicy, QFrame, \
        QHBoxLayout
except ImportError:
    try:
        from PyQt4.QtCore import Qt, QTimer, QSize, QEvent
        from PyQt4.QtGui import QWidget, QApplication, QSizePolicy, QFrame, \
            QHBoxLayout
    except ImportError:
        try:
            from PySide.QtCore import Qt, QTimer, QSize, QEvent
            from PySide.QtGui import QWidget, QApplication, QSizePolicy, \
                QFrame, QHBoxLayout
        except ImportError:
            raise ImportError("Cannot load either PyQt4, PyQt5 or PySide")

import vtk


class QVTKRenderWindowInteractor(QWidget):

    """ A QVTKRenderWindowInteractor for Python and Qt.  Uses a
    vtkGenericRenderWindowInteractor to handle the interactions.  Use
    GetRenderWindow() to get the vtkRenderWindow.  Create with the
    keyword stereo=1 in order to generate a stereo-capable window.

    Caution: Retina display handling requires a parent for the widget,
    preferably a tight fitting frame.

    The user interface is summarized in vtkInteractorStyle.h:

    - Keypress j / Keypress t: toggle between joystick (position
    sensitive) and trackball (motion sensitive) styles. In joystick
    style, motion occurs continuously as long as a mouse button is
    pressed. In trackball style, motion occurs when the mouse button
    is pressed and the mouse pointer moves.

    - Keypress c / Keypress o: toggle between camera and object
    (actor) modes. In camera mode, mouse events affect the camera
    position and focal point. In object mode, mouse events affect
    the actor that is under the mouse pointer.

    - Button 1: rotate the camera around its focal point (if camera
    mode) or rotate the actor around its origin (if actor mode). The
    rotation is in the direction defined from the center of the
    renderer's viewport towards the mouse position. In joystick mode,
    the magnitude of the rotation is determined by the distance the
    mouse is from the center of the render window.

    - Button 2: pan the camera (if camera mode) or translate the actor
    (if object mode). In joystick mode, the direction of pan or
    translation is from the center of the viewport towards the mouse
    position. In trackball mode, the direction of motion is the
    direction the mouse moves. (Note: with 2-button mice, pan is
    defined as <Shift>-Button 1.)

    - Button 3: zoom the camera (if camera mode) or scale the actor
    (if object mode). Zoom in/increase scale if the mouse position is
    in the top half of the viewport; zoom out/decrease scale if the
    mouse position is in the bottom half. In joystick mode, the amount
    of zoom is controlled by the distance of the mouse pointer from
    the horizontal centerline of the window.

    - Keypress 3: toggle the render window into and out of stereo
    mode.  By default, red-blue stereo pairs are created. Some systems
    support Crystal Eyes LCD stereo glasses; you have to invoke
    SetStereoTypeToCrystalEyes() on the rendering window.  Note: to
    use stereo you also need to pass a stereo=1 keyword argument to
    the constructor.

    - Keypress e: exit the application.

    - Keypress f: fly to the picked point

    - Keypress p: perform a pick operation. The render window interactor
    has an internal instance of vtkCellPicker that it uses to pick.

    - Keypress r: reset the camera view along the current view
    direction. Centers the actors and moves the camera so that all actors
    are visible.

    - Keypress s: modify the representation of all actors so that they
    are surfaces.

    - Keypress u: invoke the user-defined function. Typically, this
    keypress will bring up an interactor that you can type commands in.

    - Keypress w: modify the representation of all actors so that they
    are wireframe.
    """

    # Map between VTK and Qt cursors.
    _CURSOR_MAP = {
        0:  Qt.ArrowCursor,          # VTK_CURSOR_DEFAULT
        1:  Qt.ArrowCursor,          # VTK_CURSOR_ARROW
        2:  Qt.SizeBDiagCursor,      # VTK_CURSOR_SIZENE
        3:  Qt.SizeFDiagCursor,      # VTK_CURSOR_SIZENWSE
        4:  Qt.SizeBDiagCursor,      # VTK_CURSOR_SIZESW
        5:  Qt.SizeFDiagCursor,      # VTK_CURSOR_SIZESE
        6:  Qt.SizeVerCursor,        # VTK_CURSOR_SIZENS
        7:  Qt.SizeHorCursor,        # VTK_CURSOR_SIZEWE
        8:  Qt.SizeAllCursor,        # VTK_CURSOR_SIZEALL
        9:  Qt.PointingHandCursor,   # VTK_CURSOR_HAND
        10: Qt.CrossCursor,          # VTK_CURSOR_CROSSHAIR
    }

    def __init__(self, parent=None, wflags=Qt.WindowFlags(), **kw):
        # the current button
        self._ActiveButton = Qt.NoButton

        # private attributes
        self.__saveX = 0
        self.__saveY = 0
        self.__saveModifiers = Qt.NoModifier
        self.__saveButtons = Qt.NoButton

        # do special handling of some keywords:
        # stereo, rw

        stereo = 0

        if 'stereo' in kw:
            if kw['stereo']:
                stereo = 1

        rw = None

        if 'rw' in kw:
            rw = kw['rw']

        # create qt-level widget
        QWidget.__init__(self, parent, wflags | Qt.MSWindowsOwnDC)

        # check if retina display, or use 'retina' argument
        try:
            self.ratio = self.devicePixelRatio()
        except:
            self.ratio = 1
        if 'retina' in kw:
            if kw['retina']:
                self.ratio = 2
        if parent is None:
            self.ratio = 1

        # to avoid strange boundary artefacts
        # if self.ratio > 1:
        #    self.ratio *= 1.01
        # unfortunately this causes problems with PNG writing

        if rw:  # user-supplied render window
            self._RenderWindow = rw
        else:
            self._RenderWindow = vtk.vtkRenderWindow()

        WId = self.winId()

        if type(WId).__name__ == 'PyCObject':
            from ctypes import pythonapi, c_void_p, py_object

            pythonapi.PyCObject_AsVoidPtr.restype = c_void_p
            pythonapi.PyCObject_AsVoidPtr.argtypes = [py_object]

            WId = pythonapi.PyCObject_AsVoidPtr(WId)

        self._RenderWindow.SetWindowInfo(str(int(WId)))

        if stereo:  # stereo mode
            self._RenderWindow.StereoCapableWindowOn()
            self._RenderWindow.SetStereoTypeToCrystalEyes()

        if 'iren' in kw:
            self._Iren = kw['iren']
        else:
            self._Iren = vtk.vtkGenericRenderWindowInteractor()
            self._Iren.SetRenderWindow(self._RenderWindow)

        # do all the necessary qt setup
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        self.setAttribute(Qt.WA_PaintOnScreen)
        self.setMouseTracking(True)  # get all mouse events
        self.setFocusPolicy(Qt.WheelFocus)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding,
                                       QSizePolicy.Expanding))

        self._Timer = QTimer(self)
        self._Timer.timeout.connect(self.TimerEvent)

        self._Iren.AddObserver('CreateTimerEvent', self.CreateTimer)
        self._Iren.AddObserver('DestroyTimerEvent', self.DestroyTimer)
        self._Iren.GetRenderWindow().AddObserver('CursorChangedEvent',
                                                 self.CursorChangedEvent)

        # Create a hidden child widget and connect its destroyed signal to its
        # parent ``Finalize`` slot. The hidden children will be destroyed before
        # its parent thus allowing cleanup of VTK elements.
        self._hidden = QWidget(self)
        self._hidden.hide()
        self._hidden.destroyed.connect(self.Finalize)

    def __getattr__(self, attr):
        """ Make the object behave like a vtkGenericRenderWindowInteractor. """
        if attr == '__vtk__':
            return lambda t=self._Iren: t
        elif hasattr(self._Iren, attr):
            return getattr(self._Iren, attr)
        else:
            raise AttributeError(self.__class__.__name__ +
                                 " has no attribute named " + attr)

    def Finalize(self):
        """ Call internal cleanup method on VTK objects. """
        self._RenderWindow.Finalize()

    def CreateTimer(self, obj, evt):
        self._Timer.start(10)

    def DestroyTimer(self, obj, evt):
        self._Timer.stop()
        return 1

    def TimerEvent(self):
        self._Iren.TimerEvent()

    def CursorChangedEvent(self, obj, evt):
        """Called when the CursorChangedEvent fires on the render window."""
        # This indirection is needed since when the event fires, the current
        # cursor is not yet set so we defer this by which time the current
        # cursor should have been set.
        QTimer.singleShot(0, self.ShowCursor)

    def HideCursor(self):
        """Hide the cursor."""
        self.setCursor(Qt.BlankCursor)

    def ShowCursor(self):
        """Show the cursor."""
        vtk_cursor = self._Iren.GetRenderWindow().GetCurrentCursor()
        qt_cursor = self._CURSOR_MAP.get(vtk_cursor, Qt.ArrowCursor)
        self.setCursor(qt_cursor)

    def closeEvent(self, evt):
        self.Finalize()

    def sizeHint(self):
        return QSize(400, 400)

    def paintEngine(self):
        return None

    def paintEvent(self, ev):
        self._Iren.Render()

    def resizeEvent(self, ev):
        """
        Double the size on retina displays.

        This is not the right way to do it, but this works for framed widgets.
        We also need to modify all mouse events to adjust the interactor's
        center (e.g. for joystick mode).
        """
        w = self.width()
        h = self.height()
        if self.ratio > 1 and self.parent() is not None and \
                w <= self.parent().width():
            self.resize(self.ratio*self.size())
        self._RenderWindow.SetSize(self.ratio*w, self.ratio*h)
        self._Iren.SetSize(self.ratio*w, self.ratio*h)
        self._Iren.ConfigureEvent()
        self.update()

    def _GetCtrlShift(self, ev):
        ctrl = shift = False

        if hasattr(ev, 'modifiers'):
            if ev.modifiers() & Qt.ShiftModifier:
                shift = True
            if ev.modifiers() & Qt.ControlModifier:
                ctrl = True
        else:
            if self.__saveModifiers & Qt.ShiftModifier:
                shift = True
            if self.__saveModifiers & Qt.ControlModifier:
                ctrl = True

        return ctrl, shift

    def SetEventInformationFlipY(self, x, y, *args):
        """ Cheat about mouse position on retina displays. """
        self._Iren.SetEventInformationFlipY(self.ratio*x, self.ratio*y, *args)

    def enterEvent(self, ev):
        ctrl, shift = self._GetCtrlShift(ev)
        self.SetEventInformationFlipY(self.__saveX, self.__saveY,
                                      ctrl, shift, chr(0), 0, None)
        self._Iren.EnterEvent()

    def leaveEvent(self, ev):
        ctrl, shift = self._GetCtrlShift(ev)
        self.SetEventInformationFlipY(self.__saveX, self.__saveY,
                                      ctrl, shift, chr(0), 0, None)
        self._Iren.LeaveEvent()

    def mousePressEvent(self, ev):
        ctrl, shift = self._GetCtrlShift(ev)
        repeat = 0
        if ev.type() == QEvent.MouseButtonDblClick:
            repeat = 1
        self.SetEventInformationFlipY(ev.x(), ev.y(),
                                      ctrl, shift, chr(0), repeat, None)

        self._ActiveButton = ev.button()

        if self._ActiveButton == Qt.LeftButton:
            self._Iren.LeftButtonPressEvent()
        elif self._ActiveButton == Qt.RightButton:
            self._Iren.RightButtonPressEvent()
        elif self._ActiveButton == Qt.MidButton:
            self._Iren.MiddleButtonPressEvent()

    def mouseReleaseEvent(self, ev):
        ctrl, shift = self._GetCtrlShift(ev)
        self.SetEventInformationFlipY(ev.x(), ev.y(),
                                      ctrl, shift, chr(0), 0, None)

        if self._ActiveButton == Qt.LeftButton:
            self._Iren.LeftButtonReleaseEvent()
        elif self._ActiveButton == Qt.RightButton:
            self._Iren.RightButtonReleaseEvent()
        elif self._ActiveButton == Qt.MidButton:
            self._Iren.MiddleButtonReleaseEvent()

    def mouseMoveEvent(self, ev):
        self.__saveModifiers = ev.modifiers()
        self.__saveButtons = ev.buttons()
        self.__saveX = ev.x()
        self.__saveY = ev.y()

        ctrl, shift = self._GetCtrlShift(ev)
        self.SetEventInformationFlipY(ev.x(), ev.y(),
                                      ctrl, shift, chr(0), 0, None)
        self._Iren.MouseMoveEvent()

    def keyPressEvent(self, ev):
        ctrl, shift = self._GetCtrlShift(ev)
        if ev.key() < 256:
            key = str(ev.text())
        else:
            key = chr(0)

        self.SetEventInformationFlipY(self.__saveX, self.__saveY,
                                        ctrl, shift, key, 0, None)
        self._Iren.KeyPressEvent()
        self._Iren.CharEvent()

    def keyReleaseEvent(self, ev):
        ctrl, shift = self._GetCtrlShift(ev)
        if ev.key() < 256:
            key = chr(ev.key())
        else:
            key = chr(0)

        self.SetEventInformationFlipY(self.__saveX, self.__saveY,
                                      ctrl, shift, key, 0, None)
        self._Iren.KeyReleaseEvent()

    def wheelEvent(self, ev):
        pass
        #if ev.delta() >= 0:
        #    self._Iren.MouseWheelForwardEvent()
        #else:
        #    self._Iren.MouseWheelBackwardEvent()

    def GetRenderWindow(self):
        return self._RenderWindow

    def Render(self):
        self.update()


def QFramedVTK(**kw):
    """
    Wrap the VTK widget in a tight fitting frame.

    This ensures that the widget has a parent, and allows for retina handling.
    """
    frame = QFrame()
    widget = QVTKRenderWindowInteractor(frame, **kw)
    vl = QHBoxLayout()
    vl.setContentsMargins(0, 0, 0, 0)
    vl.addWidget(widget)
    frame.setLayout(vl)
    return frame, widget


def QVTKRenderWidgetConeExample():
    """A simple example that uses the QVTKRenderWindowInteractor class."""
    # every QT app needs an app
    app = QApplication(['QVTKRenderWindowInteractor'])

    # create the widget

    frame, widget = QFramedVTK()
    # for PyQt4, no automatic retina handling
    # frame, widget = QFramedVTK(retina=True)
    widget.Initialize()
    widget.Start()

    # if you dont want the 'q' key to exit comment this.
    widget.AddObserver("ExitEvent", lambda o, e, a=app: a.quit())

    ren = vtk.vtkRenderer()
    widget.GetRenderWindow().AddRenderer(ren)

    cone = vtk.vtkConeSource()
    cone.SetResolution(8)

    coneMapper = vtk.vtkPolyDataMapper()
    coneMapper.SetInputConnection(cone.GetOutputPort())

    coneActor = vtk.vtkActor()
    coneActor.SetMapper(coneMapper)

    ren.AddActor(coneActor)

    # show the widget
    frame.show()
    # start event processing
    app.exec_()

if __name__ == "__main__":
    QVTKRenderWidgetConeExample()
