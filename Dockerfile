# ML Factory - one image serving the API and the built UI from a single process.
#
# Stage 1 builds the React app; stage 2 is the runtime. The proxy handling, the apt proxy
# file, the WORKDIR and the uvicorn CMD follow the pattern already in use for our internal
# services, so the same corporate build environment works unchanged.

# ---------------------------------------------------------------------------
# Stage 1 - frontend build
# ---------------------------------------------------------------------------
FROM node:22-slim AS frontend

ARG http_proxy=http://10.0.139.93:8080
ARG https_proxy=http://10.0.139.93:8080
ARG HTTP_PROXY=http://10.0.139.93:8080
ARG HTTPS_PROXY=http://10.0.139.93:8080

ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}

WORKDIR /build

# Dependencies first, so a source change does not re-download the npm tree.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY frontend/ ./
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2 - API runtime, which also serves the built UI
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm

ARG http_proxy=http://10.0.139.93:8080
ARG https_proxy=http://10.0.139.93:8080
ARG HTTP_PROXY=http://10.0.139.93:8080
ARG HTTPS_PROXY=http://10.0.139.93:8080

ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
# Keep loopback off the proxy so the healthcheck and any in-container call work.
ENV no_proxy=localhost,127.0.0.1
ENV NO_PROXY=localhost,127.0.0.1

WORKDIR /app

# One place defines the proxy for apt, taken from the build argument above.
RUN if [ -n "${http_proxy}" ]; then \
        printf 'Acquire::http::Proxy "%s";\nAcquire::https::Proxy "%s";\n' \
            "${http_proxy}" "${https_proxy}" > /etc/apt/apt.conf.d/99proxy; \
    fi

# libgomp1 is required by XGBoost and CatBoost; curl backs the healthcheck.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        libgomp1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Dependencies first so application changes do not invalidate the dependency layer.
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        "fastapi>=0.115" "uvicorn[standard]>=0.30" "python-multipart>=0.0.9" \
        "pydantic>=2.9" "pydantic-settings>=2.5" "boto3>=1.35" \
        "pandas>=2.2" "numpy>=1.26" "scikit-learn>=1.5" "pyarrow>=17.0" "joblib>=1.4" \
        "xgboost>=2.1" "catboost>=1.2" "openpyxl>=3.1"

COPY ml_engine ./ml_engine
COPY backend ./backend
COPY jobs ./jobs
COPY scripts ./scripts
COPY --from=frontend /build/dist ./frontend/dist

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
# Serving the UI from the API is what makes this a single service.
ENV ML_FACTORY_STATIC_DIR=/app/frontend/dist

# The control plane never needs root; it validates requests and calls AWS.
RUN useradd --create-home --uid 1000 mlfactory && chown -R mlfactory /app
USER mlfactory

EXPOSE 8520

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8520/api/v1/health || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8520"]
