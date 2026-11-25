FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm
# Slim base image: the universal image is too large and its build runs out of
# disk while extracting its PyTorch layer. The kit only needs Python, git and
# curl; uv and Harbor are installed by postCreateCommand, Docker comes from
# the docker-in-docker feature.
