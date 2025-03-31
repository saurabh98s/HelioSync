from setuptools import setup, find_packages

setup(
    name="federated_learning_client",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.0.0",
        "numpy>=1.19.0",
        "requests>=2.25.0"
    ],
    entry_points={
        'console_scripts': [
            'fl-client=fl_client.run_client:main',
        ],
    },
) 