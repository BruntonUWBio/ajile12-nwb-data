from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path


here = path.abspath(path.dirname(__file__))


with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

setup(
    name='brunton_lab_to_nwb',
    version='0.1.0',
    description='Convert data to nwb',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Ben Dichter',
    author_email='ben.dichter@catalystneuro.com',
    url='https://github.com/catalystneuro/brunton-lab-to-nwb',
    keywords='nwb',
    packages=find_packages(),
    package_data={'': ['template_metafile.yml']},
    include_package_data=True,
    install_requires=[
        'pynwb',
        'lazy_ops',
        'pandas',
        'ndx-events',
        'bqplot',
        ],
)