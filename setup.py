from pathlib import Path
from setuptools import setup

long_description = (Path(__file__).parent / "README.md").read_text()

setup(
    name="dkmri",
    version="0.0.5dev",
    description="Reproducible and efficient diffusion kurtosis imaging in Python.",
    url="https://github.com/kerkelae/dkmri",
    author="Leevi Kerkelä",
    author_email="leevi.kerkela@protonmail.com",
    license="MIT",
    packages=["dkmri"],
    install_requires=["jax", "jaxlib", "nibabel", "numba", "numpy", "sklearn"],
    scripts=["dkmri/dkmri.py"],
    long_description=long_description,
    long_description_content_type="text/markdown",
)
