#! /usr/bin/env python
# -*- coding: utf-8 -*-

from distutils.core import setup
from setuptools import setup, find_packages

setup(name='lda',
      version='1.0',
      include_package_data=True,
      packages=find_packages()
      #package_dir={'lda': './lda'}
      #packages = ['lda', 'lda.corpus']
      )
