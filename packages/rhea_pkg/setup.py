# setup.py
from setuptools import setup, find_packages

setup(
    name='rhea',
    version='66',
    packages=find_packages(),
    install_requires=[
        'PyYAML'
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
