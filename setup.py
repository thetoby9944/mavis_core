from distutils.core import setup
from pathlib import Path

from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

root_folder = "mavis"
static_folders = ["assets", ".streamlit"]

setup(
    name='mavis_core',
    version='0.1.0',
    description='Mavis Platform',
    scripts=[str((Path(root_folder) / 'scripts' / 'mavis.py'))],
    # url='https://github.com/shuds13/pyexample',
    author='Tobias Schiele',
    author_email='tobias.schiele',
    license='BSD 2-clause',
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
    include_dirs=[str(Path(root_folder) / folder) for folder in static_folders],
    package_data={
        folder: [
            str(file.relative_to("."))
            for file in (Path(root_folder) / folder).rglob("*")
            if file.is_file()
        ]
        for folder in static_folders
    },
    include_package_data=True,
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux :: Windows',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    entry_points={
        'console_scripts': [
            'mavis = mavis.app:run']
    }

)
