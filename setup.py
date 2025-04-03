from setuptools import setup

setup(
    name="crossai-ts",
    version="0.0.0.2",
    packages=[
        "caits",
        "caits.ai",
        "caits.ai.nn1d",
        "caits.ai.nn2d",
        "caits.augmentation",
        "caits.core",
        "caits.dataset",
        "caits.eda",
        "caits.experimental",
        "caits.fe",
        "caits.fe.core_spectrum",
        "caits.loading",
        "caits.performance",
        "caits.resources_handling",
        "caits.transformers"
    ],
    url="https://github.com/AIoT-Group-UoP/crossai-ts",
    license="Apache License 2.0",
    author="Pantelis Tzamalis, George Kontogiannis",
    author_email="tzamalis@ceid.upatras.gr",
    description="An open-source Python library for developing "
                "end-to-end AI pipelines for Time Series Analysis",
    install_requires=[
        "tensorflow==2.14.0",
        "tensorflow-metal==1.1.0; platform_system=='Darwin'",
        "pandas==2.2.0",
        "pyarrow==15.0.2",
        "scipy==1.12.0",
        "scikit-learn==1.4.0",
        "seaborn>=0.12.2",
        "soundfile==0.12.1",
        "tsaug==0.2.1",
        "resampy==0.4.2",
        "samplerate==0.2.1; platform_system=='Darwin'",
        "samplerate==0.1.0; platform_system=='Linux'",
        "soxr==0.3.7",
        "pyyaml==6.0.1",
        "boto3==1.29.2",
        "tqdm==4.66.2"
    ],
    python_requires=">=3.8"
)
