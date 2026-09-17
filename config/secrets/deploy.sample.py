"""ML Factory deployment settings — SAMPLE.

    cp config/secrets/deploy.sample.py config/secrets/deploy.py
    # then fill in the values below

This is a different file from ``config/secrets/config.py``, and used at a different time.
``config.py`` is what the running platform reads, continuously, from the corporate server.
This file is read once, by ``scripts/deploy_aws.py`` (or ``scripts/deploy_aws.sh``, which
loads it too) when you run ``make deploy-aws`` or ``make deploy-aws-nocli`` — the one-time step
that creates the AWS resources the platform then runs against. Nothing here is read by the
platform itself.

``config/secrets/deploy.py`` is listed in .gitignore and must never be committed. This sample
carries no real values and is tracked on purpose, so the shape of the file is documented.

An environment variable of the same name still wins over anything set here, so a value can be
overridden for one invocation (CI, say) without editing the file. Anything left as ``None``
falls through to the environment the same way as if the name were simply absent — you do not
have to delete a line to leave it unset.
"""

# ---------------------------------------------------------------------------
# Deployer AWS credentials — the identity that runs `make deploy-aws`, NOT the platform's own
# runtime role. See infrastructure/iam/deployer_policy.json for what it needs.
# ---------------------------------------------------------------------------
# PREFERRED: leave all four of these as None and rely on ~/.aws or an attached role.
AWS_ACCESS_KEY_ID = None  # "AKIA..."
AWS_SECRET_ACCESS_KEY = None  # "..."
AWS_SESSION_TOKEN = None  # only for temporary (STS) credentials

# Assume this role to deploy, instead of using the keys/chain above directly. That identity
# then only needs sts:AssumeRole on this ARN; the role itself needs deployer_policy.json.
DEPLOYER_ROLE_ARN = None  # "arn:aws:iam::123456789012:role/MlFactoryDeployer"

# ---------------------------------------------------------------------------
# Required
# ---------------------------------------------------------------------------
AWS_REGION = "eu-central-1"
AWS_ACCOUNT_ID = ""  # "123456789012"
# The stack CREATES this one, so the name must be free — and S3 names are global across every
# AWS account, so a generic name is probably already someone else's. Include the account id and
# region: "ml-factory-artifacts-123456789012-eu-central-1".
ARTIFACT_BUCKET = ""
APPROVED_DATA_BUCKET = ""  # an existing bucket holding approved datasets
DEFINITIONS_BUCKET = ""  # an existing bucket the workflow definitions get uploaded to

# ---------------------------------------------------------------------------
# Optional
# ---------------------------------------------------------------------------
APPROVED_DATA_PREFIX = "curated"
IMAGE_TAG = None  # defaults to the short git SHA of this checkout
STACK_NAME = "ml-factory"

# Supply an existing, already-reviewed role instead of having the stack create one. Leave a
# line as None to have that role created as usual — any mix of created and supplied is fine.
# A supplied role must already carry the matching infrastructure/iam/*_role_policy.json; the
# stack references it, it never attaches a policy to a role it did not create.
EXISTING_BACKEND_ROLE_ARN = None  # "arn:aws:iam::123456789012:role/MlFactoryBackendRole"
EXISTING_WORKFLOW_ROLE_ARN = None  # "arn:aws:iam::123456789012:role/MlFactoryWorkflowRole"
EXISTING_JOB_ROLE_ARN = None  # "arn:aws:iam::123456789012:role/MlFactoryJobRole"

# Set to "1" when Docker and this script run on two different machines (e.g. Docker is only
# reachable through a separate shell). Skips the build/push here and prints the commands to
# run on the Docker-capable host instead; the rest of the deploy continues normally.
SKIP_IMAGE_BUILD = None  # "1"
