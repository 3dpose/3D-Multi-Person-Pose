from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

setup(
	name = 'render_core',
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("render_core_cython",
                 sources=["render_core_cython.pyx", "render_core.cpp"],
                 language='c++',
                 include_dirs=[numpy.get_include()])],
)