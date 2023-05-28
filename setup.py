import pathlib
from setuptools import setup
from setuptools import find_packages
from pkg_resources import parse_requirements

with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in parse_requirements(requirements_txt)
    ]

setup(
    name='src',
    packages=find_packages(),
    install_requires=install_requires,
    version='0.1.0',
    description='The analysis of taste function of healthy controls and patients.',
    author='Lala Chaimae Naciri (University of Cagliari)',
    license='General Public License',
)
