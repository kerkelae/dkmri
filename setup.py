from setuptools import setup

setup(
    name="dkmri",
    version="0.0.1",
    description="Reproducible and efficient diffusion kurtosis imaging in Python.",
    url="https://github.com/kerkelae/dkmri",
    author="Leevi Kerkelä",
    author_email="leevi.kerkela@protonmail.com",
    license="MIT",
    packages=["dkmri"],
    install_requires=["jax", "jaxlib", "nibabel", "numpy",],
    scripts=["dkmri/dkmri.py"],
)
