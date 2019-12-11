from setuptools import setup, find_packages


with open("README.md", "r") as f:
    long_description = f.read()


with open("micro_cat_29/__init__.py", "r") as f:
    init = f.readlines()

for line in init:
    if '__author__' in line:
        __author__ = line.split("'")[-2]
    if '__email__' in line:
        __email__ = line.split("'")[-2]
    if '__version__' in line:
        __version__ = line.split("'")[-2]


setup(
    name='micro_cat_29',
    version='0.01',
    author='Niko McCarty',
    author_email='nmccarty@caltech.edu',
    description='A package for analyzing microtubule catastrophe.',
    long_description=long_description,
    long_description_content_type='ext/markdown',
    packages=find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
)