FROM python:3.12-slim-trixie

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates
# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh
# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh
# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

RUN mkdir -p /app/src
WORKDIR /app
COPY ./best_mlp.pth /app
COPY ./best_model.pth /app
COPY ./centroids.pth /app
COPY ./pyproject.toml /app
COPY ./src /app/src

ENV UV_NO_DEV=1

RUN uv sync