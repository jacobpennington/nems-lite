import codecs
import os.path
from setuptools import find_packages, setup

NAME = 'NEMS'

VERSION = '0.0.1a'

# with codecs.open('README.rst', encoding='utf-8') as f:
#     long_description = f.read()

GENERAL_REQUIRES = ['numpy', 'scipy', 'matplotlib', 'pandas']
# pycharm also requires: tornado
# TF modules require tensorflow
# GUI requires pyqt, pyqtgraph

EXTRAS_REQUIRES = {
    'docs': ['sphinx', 'sphinx_rtd_theme', 'pygments-enaml', 'nbsphinx', 
             'pandoc', 'IPython', 'sphinx_copybutton'],
    'nwb': ['allensdk'],
    'tensorflow': ['tensorflow==2.2', 'tensorboard'],
    'tests': ['pytest', 'pytest-benchmark'],
}

setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    zip_safe=True,
    author='LBHB',
    author_email='lbhb.ohsu@gmail.com',
    description='Neural Encoding Model System',
    url='http://neuralprediction.org',
    install_requires=GENERAL_REQUIRES,
    extras_require=EXTRAS_REQUIRES,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ],
)