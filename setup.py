from setuptools import setup, find_packages

setup(
    name="gts_engine",
    version="0.1.0",
    description="gts_engine",
    long_description="git_engine development suite: a powerful NLU training system",
    license="MIT Licence",
    url="https://idea.edu.cn",
    author="pankunhao",
    author_email="pankunhao@gmail.com",

    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    exclude_package_date={'':['.gitignore']},
    install_requires=[
        "fastapi==0.85.1",
        "huggingface-hub==0.10.1",
        "Jinja2==3.1.2",
        "joblib==1.2.0",
        "numpy==1.23.4",
        "pandas==1.5.1",
        "Pillow==9.2.0",
        "pyarrow==9.0.0",
        "pytorch-lightning==1.5.10",
        "requests==2.28.1",
        "scikit-learn==1.1.2",
        "scipy==1.9.3",
        "six==1.16.0",
        "sklearn",
        "tensorboard==2.8.0",
        "tqdm==4.64.1",
        "transformers==4.18.0",
        "uvicorn==0.19.0",
    ],

    scripts=[],
)