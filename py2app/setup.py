""" setuptools file for py2app. """
from setuptools import setup
import sys
sys.setrecursionlimit(1500)

# necessary modules
include = ['sip', 'PyQt5', 'PyQt5.QtCore', 'PyQt5.QtGui', 'PyQt5.QtWidgets',
           'bibtexparser', 'billiard']
# list of all modules to exclude
import pkgutil
exclude = set(tup[1] for tup in pkgutil.iter_modules())
# unnecessary PyQt5 pieces
import PyQt5
exPyQt = set(tup[1] for tup in pkgutil.iter_modules(path=PyQt5.__path__,
                                                    prefix=PyQt5.__name__+'.'))
exclude.update(exPyQt)
exclude.difference_update(include)

APP = ['py2app/start.py']
OPTIONS = {'argv_emulation': False,
           'includes': include,
           'excludes': exclude,
           'dylib_excludes': ['QtCLucene.framework', 'QtDBus.framework',
                              'QtNetwork.framework', 'QtQml.framework',
                              'QtQuick.framework', 'QtPrintSupport.framework'],
           'semi_standalone': True,
           'site_packages': False,
           'qt_plugins': ['platforms/libqcocoa.dylib',
                          # 'accessible/libqtaccessiblewidgets.dylib'
                          ],
           'resources': ['eigenvalues.py', 'viper3d.py', 'domains.py',
                         'boundary.py', 'longcalc.py', 'paramchecker.py',
                         'qvtk.py', 'solver.py', 'transforms.py',
                         'tools.py', 'solutiontab.py',
                         'eigGUI5.py', 'solTab5.py',
                         'res', 'res',
                         'res', 'res',
                         'res', 'config.py'],
           'iconfile': 'res/eig.icns',

           }

setup(
    app=APP,
    name="Eigenvalues",
    version="0.1a",
    author="Bartlomiej Siudeja",
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
