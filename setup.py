from setuptools import find_packages, setup

with open("README.md") as fh:
    long_description = fh.read()


def read_requirements_file(filepath):
    with open(filepath) as fin:
        requirements = fin.read()
    return requirements


REQUIRED_PACKAGES = read_requirements_file("requirements.txt")

setup(
    name="gts_engine",
    version="0.1.4",
    description="git_engine development suite: a powerful NLU training system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache 2.0",
    url="https://github.com/IDEA-CCNL/GTS-Engine",
    author="pankunhao",
    author_email="pankunhao@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    exclude_package_date={'': ['.gitignore']},
    install_requires=REQUIRED_PACKAGES,
    scripts=[],
    entry_points={
        'console_scripts': [
            'gts_engine_service = gts_engine.gts_engine_service:main',
            'gts_engine_train = gts_engine.gts_engine_train:main',
            'gts_engine_inference = gts_engine.gts_engine_inference:main'
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
