from setuptools import setup
import re

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

VERSIONFILE = "finmarkets/_version.py"
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
    description='A collection of financial tools developed during Financial Markets course at UniSi.',
    url='https://github.com/matteosan1/finmarkets',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    #requires-python = ">=3.7"
    author = 'Matteo Sani',
    author_email='matteo.sani@mpscapitalservices.it',
    maintainer = 'Matteo Sani',
    maintainer_email='matteo.sani@mpscapitalservices.it',
    license = "LICENSE.txt",
    keywords = ["financial markets", "derivatives"],
    packages=['finmarkets'],
    #py_modules=['finmarkets'],
    install_requires=['numpy',
                      'scipy',
                      'tensorflow',
                      'pickle],
    #dependencies = ["peppercorn"]

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',  
        'Programming Language :: Python :: 3',
        'Natural Language :: English',
        'Topic :: Office/Business :: Financial'
    ],
)
