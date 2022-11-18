from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="gts_engine",
    version="0.1.2",
    description="git_engine development suite: a powerful NLU training system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT Licence",
    url="https://github.com/IDEA-CCNL/GTS-Engine",
    author="pankunhao",
    author_email="pankunhao@gmail.com",

    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    exclude_package_date={'':['.gitignore']},
    install_requires=[
        "fastapi==0.86.0",
        "numpy==1.22.3",
        "psutil==5.8.0",
        "pydantic==1.10.2",
        "pynvml==11.0.0",
        "pytorch_lightning==1.6.2",
        "scikit_learn==1.1.3",
        "setuptools==58.0.4",
        "starlette==0.20.4",
        "torch==1.11.0",
        "tqdm==4.62.3",
        "transformers==4.18.0",
        "uvicorn==0.19.0",
    ],

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