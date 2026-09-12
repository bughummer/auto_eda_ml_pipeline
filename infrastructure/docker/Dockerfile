# One image for every ML Factory job.
#
# SageMaker Processing supplies the entrypoint explicitly (python -m jobs.<stage>.main).
# SageMaker Training runs the image with the argument "train" and passes parameters in
# /opt/ml/input/config/hyperparameters.json, which entrypoint.sh translates into CLI flags.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    ML_FACTORY_LOG_LEVEL=INFO

# libgomp is required by XGBoost and CatBoost; jq parses the SageMaker hyperparameter file.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 jq \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml-factory

# Dependencies first so application changes do not invalidate the dependency layer.
COPY pyproject.toml ./
RUN pip install --upgrade pip setuptools wheel && pip install "." --no-deps \
    && pip install \
        "pydantic>=2.9" "pandas>=2.2" "numpy>=1.26" "scikit-learn>=1.5" "pyarrow>=17.0" \
        "joblib>=1.4" "boto3>=1.35" "xgboost>=2.1" "catboost>=1.2"

COPY ml_engine ./ml_engine
COPY jobs ./jobs
COPY infrastructure/docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV PYTHONPATH=/opt/ml-factory

# Run as a non-root user; the job only needs to read its inputs and write its artifacts.
RUN useradd --create-home --uid 1000 mlfactory && chown -R mlfactory /opt/ml-factory
USER mlfactory

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
