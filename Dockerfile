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

# By default the embedding model is pulled at first use, not baked into the image.
# Bake runs with the default cache path on purpose: if it used FASTEMBED_CACHE_PATH below,
# a runtime volume mounted over /data would shadow the baked model and force a re-download.
# Set --build-arg BAKE_EMBED_MODEL=true for air-gapped/no-download cold starts (adds ~1.3GB).
ARG BAKE_EMBED_MODEL=false
ARG EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
RUN if [ "$BAKE_EMBED_MODEL" = "true" ]; then \
      python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='$EMBEDDING_MODEL')"; \
    fi

# Runtime downloads land on the /data volume (mounted by compose and Helm), so restarts
# don't re-pull. Without a /data mount the cache is ephemeral.
ENV FASTEMBED_CACHE_PATH=/data/fastembed

COPY litestream.yml /etc/litestream.yml
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
