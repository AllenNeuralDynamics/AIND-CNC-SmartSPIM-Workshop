# hash:sha256:b58ca2caf619d9bac8e834fe171fbcb8a9786545efed6fe2b4f6b2c3f6450aae
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-access-api[docdb]==0.16.1 \
    dask==2023.5.0 \
    ipywidgets==8.1.5 \
    matplotlib==3.9.2 \
    pandas==1.5.3 \
    scikit-learn==1.4.0 \
    seaborn==0.12.2 \
    zarr==2.14.2

COPY postInstall /
RUN /postInstall
