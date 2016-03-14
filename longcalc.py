"""Progress dialog with multiprocessing worker handling and monitoring."""

from billiard import Queue, Process

try:
    from PyQt5.QtWidgets import QProgressDialog
    from PyQt5.QtCore import QTimer
except:
    from PyQt4.QtGui import QProgressDialog
    from PyQt4.QtCore import QTimer


class LongCalculation(QProgressDialog):

    """
    Multiprocessing based worker for mesh and eigenvalue calculations.

    This is necessary to make sure GUI is not blocked while mesh is built,
    or when eigenvalue calculations are performed.

    Transformations do not need as much time, unless there is one implemented
    without numpy vectorized coordinate calculations.
    """

    res = None

    def __init__(self, fun, args, postprocess, job):
        """ Build multiprocessing queues and start worker. """
        super(LongCalculation, self).__init__(job, "Cancel", 0, 0)
        self.setModal(True)
        self.input = Queue()
        self.output = Queue()
        self.input.put((fun, args, postprocess))
        self.proc = Process(target=worker, args=(self.input, self.output))
        self.proc.start()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(10)

    def update(self):
        """ Check if worker is done, and close dialog. """
        try:
            out = self.output.get(block=False)
            if isinstance(out, basestring):
                self.setLabelText(out)
                return
            if out is None:
                self.done(0)
                return
            self.res = out
            self.timer.stop()
            self.proc.join()
            del self.proc
            self.done(1)
        except:
            pass

    def cleanUp(self):
        """ Kill the running processes if cancelled/failed. """
        if self.proc:
            while self.proc.is_alive():
                self.proc.terminate()
            del self.proc
        self.timer.stop()


def worker(input, output):
    """
    Multiprocessing worker function.

    Gets function and arguments from input queue, evaluates and
    puts in output queue.

    Domains have __call__ method hence can be used as functions.
    """
    fun, args, postprocess = input.get()
    result = fun(*args, monitor=output)
    # Mesh is not picklable
    # but cell arrays and coordinates are
    # postprocessing function gets the arrays from the mesh
    output.put(postprocess(result))


def pickle_mesh(mesh):
    """ Get data from mesh which can be pickled. """
    return [mesh.cells(), mesh.coordinates()]


def pickle_solutions(solutions):
    """ Get arrays from eigenfunctions. """
    if solutions is None:
        return None
    eigv = solutions[0]
    eigf = solutions[1]
    for i in range(len(eigf)):
        eigf[i] = eigf[i].vector().array()
    solutions[0] = eigv
    solutions[1] = eigf
    return solutions
