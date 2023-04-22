from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.1.3'
DESCRIPTION = 'MGD_Outliers is a Pypyththon package for identifying and visualizing outliers in a DataFrame.'


# Setting up
setup(
    name="MGD_Outliers",
    version=VERSION,
    author="Mayur Dushetwar",
    author_email="<mdushetwar@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    license='Apache License 2.0',
    install_requires=['numpy', 'pandas', 'matplotlib', 'seaborn', 'jinja2'],
    keywords=['python', 'dataframe', 'outlier', 'outliers', 'inter quartile range', 'plot outliers', 'count outliers', 'data processing', 'drop outliers'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    python_requires= '>= 3.9.7',
    py_modules=['MGD_Outliers'],
    package_dir={'':'utils'}

)