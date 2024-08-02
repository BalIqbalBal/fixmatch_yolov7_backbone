# setup.py
from setuptools import setup, find_packages

setup(
    name='polyaxon_schemas',
    version='66',
    packages=find_packages(),
    install_requires=[
        "hestia==0.3.1",
        "Jinja2>=3.1.2",
        "marshmallow",
        "numpy>=1.15.2",
        "python-dateutil>=2.7.3",
        "pytz>=2018.9",
        "rhea"
    ],
    entry_points={
        'console_scripts': [
            # If you have any command-line scripts, define them here
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)