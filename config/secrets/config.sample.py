"""ML Factory credentials and environment settings — SAMPLE.

    cp config/secrets/config.sample.py config/secrets/config.py
    # then fill in the values below

``config/secrets/config.py`` is listed in .gitignore and must never be committed. This sample
carries no real values and is tracked on purpose, so the shape of the file is documented.

Precedence, highest first:

    1. environment variables (ML_FACTORY_*)   — what docker-compose and systemd set
    2. the .env file
    3. this file
    4. the built-in defaults

So a value set here can still be overridden per deployment without editing the file, and
anything you leave as ``None`` simply falls through to the layer below.

Names are matched with or without the ``ML_FACTORY_`` prefix: ``AWS_REGION`` and
``ML_FACTORY_AWS_REGION`` both work. Unknown names are ignored, so you can keep notes and
helper variables here (prefix them with ``_`` to be explicit).
"""

LOG_LEVEL = "INFO"

# ---------------------------------------------------------------------------
# AWS credentials
# ---------------------------------------------------------------------------
# PREFERRED: leave all four of these as None and give the server an instance role,
# or mount ~/.aws into the container. Nothing to rotate, nothing to leak.
#
# Use static keys only where no role is available. They must be set as a pair.
AWS_ACCESS_KEY_ID = None  # "AKIA..."
AWS_SECRET_ACCESS_KEY = None  # "..."
AWS_SESSION_TOKEN = None  # only for temporary (STS) credentials

# Or name a profile from ~/.aws/credentials instead of typing keys here:
AWS_PROFILE = None  # "ml-factory"

AWS_REGION = "eu-central-1"

# ---------------------------------------------------------------------------
# AWS resources (all required)
# ---------------------------------------------------------------------------
# All of these come from the CloudFormation stack outputs; see infrastructure/README.md.
#
# Artifacts and experiment records both live in the artifact bucket. There is no database.
ARTIFACT_BUCKET = ""  # "s3://my-ml-factory-artifacts"
EDA_STATE_MACHINE_ARN = ""  # "arn:aws:states:eu-central-1:123456789012:stateMachine:ml-factory-eda"
TRAINING_STATE_MACHINE_ARN = ""  # "arn:aws:states:...:stateMachine:ml-factory-training"
KMS_KEY_ID = ""  # customer-managed key protecting the artifact bucket

# Other artifact buckets the UI may browse read-only, to see experiments that another
# environment produced. New experiments are always written to ARTIFACT_BUCKET.
ADDITIONAL_ARTIFACT_ROOTS = [
    # "s3://ml-factory-artifacts-prod",
]

# Datasets outside these prefixes are rejected before any AWS call is made.
# An empty list denies every dataset — that is deliberate, not a bug.
ALLOWED_DATASET_PREFIXES = [
    # "s3://my-approved-data/curated",
]

# ---------------------------------------------------------------------------
# Bedrock (the semantic analysis tab). Optional — the platform works without it.
# ---------------------------------------------------------------------------
BEDROCK_ENABLED = False
BEDROCK_MODEL_ID = ""  # the model id approved for use in your account
BEDROCK_MAX_TOKENS = 4096

# ---------------------------------------------------------------------------
# Web layer
# ---------------------------------------------------------------------------
# Only needed when the UI is served from a different origin than the API. The container
# serves both from one origin, so it can stay as it is.
CORS_ORIGINS = ["http://localhost:5173"]

# Upload ceiling for data dictionaries, in bytes.
MAX_DICTIONARY_UPLOAD_BYTES = 10 * 1024 * 1024

# ---------------------------------------------------------------------------
# There is deliberately nothing here for:
#   - user authentication: the corporate reverse proxy authenticates and passes the
#     identity through the X-Remote-User header
#   - the corporate proxy: HTTP_PROXY / HTTPS_PROXY / NO_PROXY are read from the
#     environment, as the rest of the estate expects
# ---------------------------------------------------------------------------
