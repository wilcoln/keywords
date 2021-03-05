import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="keywords",
    version="0.0.1",
    author="Wilfried L. Bounsi",
    author_email="wilcoln99@gmail.com",
    description="A package that provides keywords extraction datasets and methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wilcoln/keywords",
    packages=setuptools.find_packages(),
    install_requires=[
        'pandas',
        'yake',
        'spacy',
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
