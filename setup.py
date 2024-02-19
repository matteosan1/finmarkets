#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup

def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

VERSIONFILE = "src/finmarkets/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setup(
    name='finmarkets',
    version=verstr,
    license = "LICENSE.txt",
    description='A collection of financial tools developed during Financial Markets course at UniSi.',
    url='https://github.com/matteosan1/finmarkets',
    project_urls = {
        'Changelog': 'https://github.com/matteosan1/finmarkets/blob/master/CHANGELOG.rst',
        'Issue Tracker': 'https://github.com/matteosan1/finmarkets/issues',
    },
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    #requires-python = ">=3.7"
    author = 'Matteo Sani',
    author_email='matteo.sani@unisi.it',
    maintainer = 'Matteo Sani',
    maintainer_email='matteo.sani@unisi.it',
    keywords = ["financial markets", "derivatives"],
    #packages=['finmarkets'],
    packages=find_packages('src'),
    package_dir={'': 'src'},
    #py_modules=['finmarkets'],
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    #include_package_data=True,
    zip_safe=False,
    python_requires='>=3.5, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*',
    install_requires=['numpy', 'scipy',], # 'tensorflow'],
    extras_require={
        # eg:
        #   'rst': ['docutils>=0.11'],
        #   ':python_version=="2.6"': ['argparse'],
    },
#    setup_requires = ['pytest-runner',],
#    entry_points = {
#        'console_scripts': [
#            'nameless = nameless.cli:main',
#        ]
#    },
    #dependencies = ["peppercorn"]
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',  
        'Programming Language :: Python :: 3',
        'Natural Language :: English',
        'Topic :: Education',
        'Topic :: Office/Business :: Financial',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',        
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Utilities',
    ],
)

