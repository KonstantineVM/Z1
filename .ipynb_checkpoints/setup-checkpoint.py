"""
Setup script for Economic Time Series Analysis Framework
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f 
                if line.strip() and not line.startswith('#')]

# Get version
def get_version():
    version_file = os.path.join('src', '__version__.py')
    if os.path.exists(version_file):
        with open(version_file) as f:
            exec(f.read())
            return locals()['__version__']
    return '1.0.0'

setup(
    name="economic-timeseries-analysis",
    version=get_version(),
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive framework for analyzing Federal Reserve economic time series data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/economic-timeseries-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Office/Business :: Financial",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements('requirements.txt'),
    extras_require={
        'dev': read_requirements('requirements-dev.txt') if os.path.exists('requirements-dev.txt') else [],
        'docs': ['sphinx>=4.0.0', 'sphinx-rtd-theme>=0.5.0'],
        'api': ['fastapi>=0.70.0', 'uvicorn>=0.15.0'],
    },
    entry_points={
        'console_scripts': [
            'econ-cache=src.data.cache_manager:cli',
            'econ-analyze=src.cli.main:main',
        ],
    },
    include_package_data=True,
    package_data={
        'economic_timeseries_analysis': [
            'config/*.yaml',
            'config/*.yml',
            'data/.gitkeep',
            'notebooks/*.ipynb',
            'examples/*.py',
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/economic-timeseries-analysis/issues",
        "Documentation": "https://economic-timeseries-analysis.readthedocs.io/",
        "Source": "https://github.com/yourusername/economic-timeseries-analysis",
    },
    keywords=[
        'economics', 'time-series', 'federal-reserve', 'analysis',
        'machine-learning', 'forecasting', 'decomposition',
        'z1', 'flow-of-funds', 'economic-indicators'
    ],
    zip_safe=False,
)