from setuptools import setup, find_packages

setup(
    name="federated-learning-client",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests",
        "numpy",
        "torch"
    ],
    description="Federated Learning Client Package",
) 