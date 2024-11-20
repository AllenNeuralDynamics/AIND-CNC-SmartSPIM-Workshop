# hash:sha256:6f7a359040e57d8a98b19f384a9ed8cbc60a12f3ac4cb817ba2abb6461001ed1
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/py-r:python3.10.12-R4.2.3-IRkernel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-access-api[docdb]==0.16.1 \
    allensdk==2.16.2 \
    antspyx==0.4.2 \
    ccf-streamlines==1.1.4 \
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
