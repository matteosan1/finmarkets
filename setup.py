from setuptools import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='finmarkets',
    version='1.0.0',    
    description='A collection of financial tools developed during Financial Markets course at UniSi.',
    #url='https://github.com/shuds13/pyexample',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    #requires-python = ">=3.7"
    author = 'Matteo Sani',
    author_email='matteo.sani@mpscapitalservices.it',
    maintainer = 'Matteo Sani',
    maintainer_email='matteo.sani@mpscapitalservices.it',
    license = "LICENSE.txt",
    keywords = ["financial markets", "derivatives"],
    #packages=['finmarkets'],
    py_modules=['finmarkets'],
    install_requires=['numpy',
                      'scipy'],
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
