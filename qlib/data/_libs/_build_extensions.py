"""Local build helper — compile qlib's C++ Cython extensions for the current
Python interpreter, leaving ``*.pyd`` (Windows) / ``*.so`` (Unix) next to the
``.pyx`` source.

Usage (Windows):

    # Ensure MSVC env is loaded first (vcvars64.bat) — see build.bat

    python qlib/data/_libs/_build_extensions.py build_ext --inplace

After this, ``from qlib.data._libs.rolling import ...`` works for that
interpreter. Commit the resulting ``.{cp313,cp311,...}-*.pyd`` files back to
the repo so downstream consumers don't need to rebuild.
"""

from pathlib import Path

import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize


_THIS_DIR = Path(__file__).resolve().parent

# language="c++" is required at Extension declaration time — cythonize inspects
# it to decide whether to emit .c or .cpp (rolling.pyx uses libcpp.deque).
extensions = [
    Extension(
        name="qlib.data._libs.rolling",
        sources=[str(_THIS_DIR / "rolling.pyx")],
        language="c++",
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        name="qlib.data._libs.expanding",
        sources=[str(_THIS_DIR / "expanding.pyx")],
        language="c++",
        include_dirs=[numpy.get_include()],
    ),
]

setup(
    name="qlib_data_libs_local_build",
    ext_modules=cythonize(extensions, language_level=3),
)
