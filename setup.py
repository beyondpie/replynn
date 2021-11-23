import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="replynn",
    version="0.0.1",
    author="Songpeng Zu",
    author_email="szu@health.ucsd.edu",
    description="utilities for immune-related project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/beyondpie/replynn",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires = [
        'numpy',
        'sklearn',
        'torch',
        'torchvision',
        'torchaudio'
    ],
    include_package_data = True,
    packages = ['replynn'],
    package_dir = {'replynn': 'replynn'},
    package_data = {'replynn': ['data/*']},
    zip_safe = True,
)
