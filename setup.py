from setuptools import setup, find_packages

reqs = [
    "tensorflow",
    "numpy",
    "requests",
]

setup(
    name="colorbot",
    version="0.0.0",
    packages=find_packages(),

    install_requires=reqs,
)
