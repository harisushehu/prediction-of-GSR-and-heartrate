# hash:sha256:78f4eb08fcd5b89e1ec554fc515abe55f1914650d299fd21fcfdec2a11ae34d9
FROM registry.codeocean.com/codeocean/miniconda3:4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install -U --no-cache-dir \
    matplotlib==3.7.1 \
    numpy==1.24.2 \
    pandas==1.5.3 \
    pyswarms==1.3.0 \
    scikit-learn==1.2.1 \
    scipy==1.10.1 \
    tensorflow==2.11.0
