from setuptools import setup, find_packages

setup(
    name="neurotorch",
    version="0.1.0",
    packages=find_packages(),
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
)
