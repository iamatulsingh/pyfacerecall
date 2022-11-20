import setuptools
from pyfacerecall._version import __version__

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as req:
    reqs = req.read().split("\n")

setuptools.setup(
    name="pyfacerecall",
    version=__version__,
    author="Atul Singh",
    author_email="atulsingh0401@gmail.com",
    description="A Machine Learning Python library for face recognition.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iamatulsingh/pyfacerecall",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=reqs,
)
