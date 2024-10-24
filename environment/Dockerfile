# hash:sha256:e0b8d7ab5041e34ce6cfc10fbf8ef1b30648fd9f8d3d59cdce2d2253e96c160a
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

COPY postInstall /
RUN /postInstall
