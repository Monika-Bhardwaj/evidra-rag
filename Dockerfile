# EVIDRA production image.
# Security: runs as a dedicated non-root user (uid 10001); curl is installed
# while still root and only for the compose healthcheck.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin evidra

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data /app/index \
    && chown -R evidra:evidra /app

USER evidra

EXPOSE 8000 8501

ENTRYPOINT ["/bin/bash", "scripts/docker_entrypoint.sh"]
CMD ["api"]
