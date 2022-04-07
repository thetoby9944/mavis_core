from distutils.core import setup
from pathlib import Path

from setuptools import find_packages

with open("README.rst", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

with open('mavis/__init__.py') as f:
    version = f.readlines()[0].split("=")[1].strip().replace('"', "")


root_folder = "mavis"
static_folders = ["assets", ".streamlit", "scripts"]


setup(
    version=version,
    name='mavis_core',
    description='Mavis - Machine Learning and Computer Vision. Turn code into applications.',
    long_description=long_description,
    scripts=[str((Path(root_folder) / 'scripts' / 'mavis_core.py'))],
    url='https://github.com/thetoby9944/mavis_core',
    author='thetoby9944',
    author_email='thetoby@web.de',
    license='AGPL v3',
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
        'License :: Free for internal use - commercial distribution prohibited',
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
