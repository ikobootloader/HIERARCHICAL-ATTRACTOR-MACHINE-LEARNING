"""Setup script for HAML."""

from setuptools import find_packages, setup


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8-sig") as fh:
    requirements = []
    for raw_line in fh:
        line = raw_line.strip()
        if not line:
            continue
        if line.lstrip("\ufeff").startswith("#"):
            continue
        requirements.append(line)


setup(
    name="haml",
    version="0.2.2",
    author="HAML Contributors",
    description="Hierarchical Attractor Machine Learning - Systeme de classification par dynamique d'attracteurs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ikobootloader/HIERARCHICAL-ATTRACTOR-MACHINE-LEARNING",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=7.0.0", "black>=22.0.0", "flake8>=4.0.0"],
        "adjoint": ["torchdiffeq>=0.2.3"],
    },
)
