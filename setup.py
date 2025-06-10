from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="clip_model",
    version="0.1.0",
    author="CLIP Model Team",
    description="A package for using CLIP models for image-text similarity",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/clip_model",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
) 