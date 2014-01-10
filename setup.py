from distutils.core import setup, Extension
try:
    import numpy
except ImportError:
    raise SystemExit('requires NumPy version > 1.6.0') 


INC_DIR = ['$HOME/include',
           '/usr/include',
           '/usr/local/include',
           numpy.get_include()]
LIB_DIR = ['$HOME/lib',
           '/usr/lib64',
           '/usr/lib',
           '/usr/local/lib',]

moduleerfa = Extension('_erfa',
                    include_dirs = INC_DIR,
                    libraries = ['erfa', 'm'],
                    library_dirs = LIB_DIR,
                    extra_compile_args = ["-std=gnu99"],
                    sources = ['src/_erfamodule.c'])

setup (name = 'erfa_numpy',
       version = '2014.01.08',
       description = 'numpy wrapper for ERFA library',
       url = 'https://github.com/nirinA/erfa_numpy',
       author = 'nirinA raseliarison',
       author_email = 'nirina.raseliarison@gmail.com',
       py_modules=['erfa'],
       ext_modules = [moduleerfa, ],
       license="Public Domain")
