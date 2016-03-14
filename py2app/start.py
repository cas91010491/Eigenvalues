"""
This script executes the main application for py2app.

Lots of paths need to be fixed.
"""
import os

os.system("""
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
FEniCS_RESOURCES=/Applications/FEniCS.app/Contents/Resources
source $FEniCS_RESOURCES/share/fenics/fenics.conf
export PYTHONPATH="$RESOURCEPATH/lib/python2.7/lib-dynload:$PYTHONPATH:$RESOURCEPATH/lib/python2.7:"
# export DYLD_PRINT_LIBRARIES=1
export DYLD_FRAMEWORK_PATH="$RESOURCEPATH/../Frameworks:$DYLD_FRAMEWORK_PATH"
export DYLD_FALLBACK_LIBRARY_PATH=$DYLD_FALLBACK_LIBRARY_PATH/usr/lib:
python eigenvalues.py
""")
