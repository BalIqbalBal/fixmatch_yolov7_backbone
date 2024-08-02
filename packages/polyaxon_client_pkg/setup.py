# setup.py
from setuptools import setup, find_packages

setup(
    name='polyaxon_client',
    version='66',
    packages=find_packages(),
    install_requires=[
        "polyaxon-schemas",
        "polystores>=0.2.4",
        "psutil>=5.4.7",
        "requests>=2.20.0",
        "requests-toolbelt>=0.8.0",
        "websocket-client>=0.53.0",
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
