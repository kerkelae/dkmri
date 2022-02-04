from pathlib import Path
from setuptools import setup

long_description = (Path(__file__).parent / "README.md").read_text()

setup(
    name="dkmri",
    version="0.0.7",
    description="Reproducible and efficient diffusion kurtosis imaging in Python.",
    url="https://github.com/kerkelae/dkmri",
    author="Leevi Kerkelä",
    author_email="leevi.kerkela@protonmail.com",
    license="MIT",
    packages=["dkmri"],
    install_requires=[
        "jax==0.2.28",
        "jaxlib==0.1.76",
        "matplotlib==3.5.1",
        "nibabel==3.2.1",
        "numba==0.55.1",
        "numpy==1.21.5",
        "scikit-learn==1.0.2",
    ],
    scripts=["dkmri/dkmri.py"],
    long_description=long_description,
    long_description_content_type="text/markdown",
)
