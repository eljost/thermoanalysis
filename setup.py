#!/usr/bin/env python3

from setuptools import setup, find_packages
import sys

if sys.version_info.major < 3:
    raise SystemExit("Python 3 is required!")

setup(
    name="nicevibes",
    version="0.1",
    description="Do themochemistry analysis.",
    url="https://github.com/eljost/nicevibes",
    maintainer="Johannes Steinmetzer",
    maintainer_email="johannes.steinmetzer@uni-jena.de",
    license="GPL 3",
    platforms=["unix"],
    packages=find_packages(),
    install_requires=[
        "numpy",
        "cclib",
    ],
    entry_points={
        "console_scripts": [
            "nicevibes = nicevibes.main:run",
        ]
    },
)
