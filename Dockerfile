FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Litestream (optional backup sidecar-in-container, PRD §11)
ARG TARGETARCH
ARG LITESTREAM_VERSION=0.5.12
RUN set -eux; \
    arch="${TARGETARCH:-amd64}"; \
    case "$arch" in \
      arm64) lsarch="arm64" ;; \
      amd64) lsarch="x86_64" ;; \
      *) lsarch="$arch" ;; \
    esac; \
    curl -fsSL "https://github.com/benbjohnson/litestream/releases/download/v${LITESTREAM_VERSION}/litestream-${LITESTREAM_VERSION}-linux-${lsarch}.tar.gz" \
      | tar -xz -C /usr/local/bin litestream

COPY pyproject.toml ./
COPY app ./app
COPY migrations ./migrations

RUN pip install --no-cache-dir -e .

# Bake the embedding model into the image so cold start doesn't need a download (Risk §14.3)
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-large-en-v1.5')"

COPY litestream.yml /etc/litestream.yml
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
