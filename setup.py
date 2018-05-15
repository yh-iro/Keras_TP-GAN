from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['scikit-image>=0.13', 'Keras>=2.1.3', 'opencv-python>=3']

setup(
    name='tpgan',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My trainer application package.'
)
