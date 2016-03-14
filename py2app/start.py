"""
This script executes the main application for py2app.

Lots of paths need to be fixed.
"""
import os

path = os.path.join(os.environ['RESOURCEPATH'], 'lib', 'python2.7',
                    'lib-dynload')
frameworks = os.path.join(os.environ['RESOURCEPATH'], '..', 'Frameworks')

os.system("""
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
FEniCS_RESOURCES=/Applications/FEniCS.app/Contents/Resources
source $FEniCS_RESOURCES/share/fenics/fenics.conf
export PYTHONPATH="{}:$PYTHONPATH"
# export DYLD_PRINT_LIBRARIES=1
export DYLD_FRAMEWORK_PATH="{}:$DYLD_FRAMEWORK_PATH"
export DYLD_FALLBACK_LIBRARY_PATH=$DYLD_FALLBACK_LIBRARY_PATH/usr/lib:
python eigenvalues.py
""".format(path, frameworks))
