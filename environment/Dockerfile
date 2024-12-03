# hash:sha256:0611bd85eb546e478813b56eda20437f011cfe1a294b79745c8da8fb31044576
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/py-r:python3.10.12-R4.2.3-IRkernel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-access-api[docdb]==0.16.1 \
    antspyx==0.4.2 \
    dask==2023.5.0 \
    ipywidgets==8.1.5 \
    matplotlib==3.9.2 \
    opencv-python-headless==4.8.0.74 \
    pandas==1.5.3 \
    scikit-learn==1.4.0 \
    seaborn==0.12.2 \
    zarr==2.14.2

COPY postInstall /
RUN /postInstall
