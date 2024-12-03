# hash:sha256:8166df9fe798425f2d4aa4e51dc15f18ee876433e67d81eababe8d9dbc169d36
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-access-api[docdb]==0.16.1 \
    dask==2023.5.0 \
    ipywidgets==8.1.5 \
    lxml==5.3.0 \
    matplotlib==3.9.2 \
    pandas==1.5.3 \
    scikit-learn==1.4.0 \
    seaborn==0.12.2 \
    zarr==2.14.2

COPY postInstall /
RUN /postInstall
