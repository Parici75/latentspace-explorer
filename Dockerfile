ARG PYTHON_DOCKER_IMAGE
FROM $PYTHON_DOCKER_IMAGE AS python-base

# System env
ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    PATH="/root/.local/bin:/app:/app/.venv/bin:${PATH}"

# Python env
ENV PYTHONPATH="/root/.local/bin:/app:/app/.venv/bin${PYTHONPATH}" \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONASYNCIODEBUG=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.8

# Install compilers
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y g++ && \
    apt-get clean

# Bootstrap folders & non-root user
RUN mkdir -p /app && \
    chown 1000:1000 -R /app

WORKDIR /app

# Base Poetry layer
FROM python-base AS python-builder

# Install git
RUN apt-get update && apt-get install -y git

# Install poetry
RUN pip install "poetry==$POETRY_VERSION" --no-warn-script-location
# Requirements layer
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.in-project true && \
    poetry install --only=main --no-root --no-interaction --no-ansi && \
    poetry self add "poetry-dynamic-versioning[plugin]"


FROM python-base AS python-app

# Declare the CONTAINER_PORT ARG only at the python-app stage, because arguments declared in parent images are not available in children !!
ARG CONTAINER_PORT=8050
ARG WORKERS=4

COPY --from=python-builder /app/.venv ./.venv
COPY lse ./lse
COPY server ./server
RUN chmod +x /app/server/entrypoint.sh

# Set up a temporary dir
RUN chmod +rw /tmp
ENV TMPDIR=/tmp

# App environment
ENV WORKERS=$WORKERS
ENV HOSTNAME=0.0.0.0
ENV PORT=$CONTAINER_PORT
ENV ENABLE_HTTPS_REDIRECT=
ENV USE_JSON_LOGGING=

# Launch app
USER 1000
ENTRYPOINT ["server/entrypoint.sh"]
