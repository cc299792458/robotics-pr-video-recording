import sys
from setuptools import setup, find_packages

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))


setup(
    name='pr-vdrcd',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/cc299792458/robotics-pr-video-recording',
    license='',
    author="SU Lab",
    install_requires=[
         'sapien', 'numpy'
    ],
)
