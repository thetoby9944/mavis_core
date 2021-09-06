from distutils.core import setup
from pathlib import Path

from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

with open('mavis/__init__.py') as f:
    version = f.readlines()[0].split("=")[1].strip().replace('"', "")


root_folder = "mavis"
static_folders = ["assets", ".streamlit"]

setup(
    name='mavis_core',
    version=version,
    description='Mavis Platform',
    scripts=[str((Path(root_folder) / 'scripts' / 'mavis.py'))],
    url='https://github.com/thetoby9944/mavis_core',
    author='thetoby9944',
    author_email='thetoby@web.de',
    license='BSD 2-clause',
    packages=find_packages(),
    install_requires=required,
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
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: Free for non-commercial use',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    entry_points={
        'console_scripts': [
            'mavis = mavis.app:run'
        ]
    }

)
