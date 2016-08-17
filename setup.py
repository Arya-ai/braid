from __future__ import print_function
from setuptools import setup, find_packages
import os


desc_BRAID = """Meta-framework for faster prototyping in Deep Learning.
"""


AUTHOR = 'Arya ai'
AUTHOR_EMAIL = 'hello@arya.ai'
DOWNLOAD_URL = 'http://github.com/Arya-ai/braid'
LICENSE = 'MIT'
DESCRIPTION_BRAID = 'Deep Learning library in Tensorflow'
LONG_DESCRIPTION_BRAID = desc_BRAID

INSTALL_REQUIRES = ['numpy', 'tensorflow', 'easydict']

if os.path.isdir('braid'):
    with open('braid/__init__.py') as fid:
        for line in fid:
            if line.startswith('__version__'):
                VERSION_BRAID = line.strip().split()[-1][1:-1]
                break

    setup(name='braid',
          version=VERSION_BRAID,
          description=DESCRIPTION_BRAID,
          long_description=LONG_DESCRIPTION_BRAID,
          download_url=DOWNLOAD_URL,
          author=AUTHOR,
          author_email=AUTHOR_EMAIL,
          license=LICENSE,
          packages=find_packages(exclude=['tests',
                                          'tests.*',
                                          '*.tests',
                                          '*.tests.*']),
          install_requires=INSTALL_REQUIRES,
          zip_safe=False)
