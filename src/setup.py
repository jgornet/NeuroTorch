from setuptools import setup, find_packages

setup(
    name="neurotorch",
    version="0.1.0",
    packages=find_packages(),
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    install_requires=["numpy>=1.14.5",
"tensorboardX>=1.2",
"tifffile==0.14.0",
"h5py>=2.7.1",
"scipy>=1.1.0",
"pybind11>=2.2.3",
"psutil>=5.4.7",
"jupyter==1.0.0",
"matplotlib==2.2.3",
"pillow",
"tensorflow==1.13.1"]
)
