"""
setup.py
========
Project setup for Customer Segmentation & Retention Analysis.
Allows the project to be installed as a package locally.

Usage:
    pip install -e .    ← Install in editable/dev mode
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name             = "customer-segmentation",
    version          = "1.0.0",
    author           = "Your Name",
    author_email     = "your.email@example.com",
    description      = "End-to-End Customer Segmentation & Retention Analysis ML Pipeline",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url              = "https://github.com/your-username/customer-segmentation",
    packages         = find_packages(),
    python_requires  = ">=3.10",
    install_requires = requirements,
    classifiers      = [
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points = {
        "console_scripts": [
            "run-pipeline=run_pipeline:main",
        ]
    },
)