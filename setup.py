from distutils.core import setup

# Read the contents of the README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='StarNet',
    version='0.1.0',
    author='Spencer Bialek',
    author_email='sbialek@uvic.ca',
    packages=['starnet', ],
    license='LICENSE.txt',
    url='www.test.com',
    description='Precise stellar parameter estimation by training a deep neural network on synthetic spectra',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        "Keras>=2.2.4",
        "astropy>=3.0.5",
        "h5py==2.7.1",
        "pysynphot>=0.9.12"
    ],
)