from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, "r", encoding="utf-8") as fh:
        return fh.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="ergodicity",
    version="0.3.19",
    author="Ihor Kendiukhov",
    author_email="kendiukhov@gmail.com",
    description="A Python library for ergodicity economics and time-average analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Kendiukhov/ergodicity_library/",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=parse_requirements('requirements.txt'),
)
