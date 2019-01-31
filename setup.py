
import numpy as np
from distutils.core import setup
from Cython.Build import cythonize

DISTNAME = 'safegridoptim'
DESCRIPTION = 'Coordinate descent solver for elastic net and l1 logistic'
LONG_DESCRIPTION = open('README.md').read()
MAINTAINER = 'Eugene Ndiaye'
MAINTAINER_EMAIL = 'ndiayeeugene@gmail.com'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/EugeneNdiaye/safe_grid_search'
URL = 'https://github.com/EugeneNdiaye/safe_grid_search'
VERSION = None

setup(name='safegridoptim',
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      license=LICENSE,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url=URL,
      download_url=DOWNLOAD_URL,
      packages=['safegridoptim'],
      ext_modules=cythonize("safegridoptim/*.pyx"),
      include_dirs=[np.get_include()]
      )
