#!/usr/bin/env bash
# One-time (and on-upgrade) AWS deployment for the ML Factory.
#
# docker compose runs the control plane on this host. It cannot create AWS resources, and it
# cannot put the job image where SageMaker can pull it — SageMaker runs a container image from
# ECR, not a checkout. This script does both, in the order they have to happen.
#
#   ./scripts/deploy_aws.sh
#
# Every setting below can be an environment variable, or a line in config/secrets/deploy.py
# (copy config/secrets/deploy.sample.py) - the environment wins where both are set. Nothing
# requires `export`; a filled-in deploy.py alone is enough to run this script.
#
# Required:
#   AWS_REGION              e.g. eu-central-1
#   AWS_ACCOUNT_ID          e.g. 123456789012
#   ARTIFACT_BUCKET         bucket the stack creates for artifacts and experiment records
#   APPROVED_DATA_BUCKET    existing bucket holding approved datasets
#   DEFINITIONS_BUCKET      existing bucket the workflow definitions are uploaded to
# Optional:
#   APPROVED_DATA_PREFIX    default: curated
#   IMAGE_TAG               default: the short git SHA
#   STACK_NAME              default: ml-factory
#   DEPLOYER_ROLE_ARN            assume this role to deploy (needs sts:AssumeRole on it; the
#                                role itself needs infrastructure/iam/deployer_policy.json)
#   EXISTING_BACKEND_ROLE_ARN    use this role instead of creating MlFactoryBackendRole
#   EXISTING_WORKFLOW_ROLE_ARN   use this role instead of creating MlFactoryWorkflowRole
#   EXISTING_JOB_ROLE_ARN        use this role instead of creating MlFactoryJobRole
#   Each Existing*RoleArn role must already carry the matching policy in infrastructure/iam/
#   (backend_role_policy.json / workflow_role_policy.json / job_role_policy.json) - this script
#   does not create or modify a role you supply.
#   SKIP_IMAGE_BUILD             set (e.g. "1") when Docker and this script run on different
#                                hosts. Skips the docker login/build/push here and instead
#                                prints the exact commands to run on the Docker-capable host,
#                                then continues on to the workflow upload and the stack deploy.
set -euo pipefail

# Pull whatever config/secrets/deploy.py sets for names not already in the environment. One
# loader (deploy_aws.py's load_deploy_secrets) serves both this script and the boto3 one, so
# the file's format and precedence rules do not drift between them.
PYTHON_BIN="python3"
[ -x .venv/bin/python ] && PYTHON_BIN=".venv/bin/python"
DEPLOY_SECRETS_FILE="${ML_FACTORY_DEPLOY_SECRETS_FILE:-config/secrets/deploy.py}"
if [ -f "${DEPLOY_SECRETS_FILE}" ]; then
    eval "$("${PYTHON_BIN}" scripts/deploy_aws.py --print-shell-exports)"
fi

if [ -n "${DEPLOYER_ROLE_ARN:-}" ]; then
    echo "==> Assuming ${DEPLOYER_ROLE_ARN}"
    read -r AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN < <(
        aws sts assume-role --role-arn "${DEPLOYER_ROLE_ARN}" \
            --role-session-name ml-factory-deploy \
            --query 'Credentials.[AccessKeyId,SecretAccessKey,SessionToken]' --output text
    )
    export AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
fi

: "${AWS_REGION:?set AWS_REGION}"
: "${AWS_ACCOUNT_ID:?set AWS_ACCOUNT_ID}"
: "${ARTIFACT_BUCKET:?set ARTIFACT_BUCKET}"
: "${APPROVED_DATA_BUCKET:?set APPROVED_DATA_BUCKET}"
: "${DEFINITIONS_BUCKET:?set DEFINITIONS_BUCKET}"

APPROVED_DATA_PREFIX="${APPROVED_DATA_PREFIX:-curated}"
STACK_NAME="${STACK_NAME:-ml-factory}"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD 2>/dev/null || echo latest)}"
REPOSITORY="ml-factory-jobs"
REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_URI="${REGISTRY}/${REPOSITORY}:${IMAGE_TAG}"

echo "==> 1/4 ECR repository ${REPOSITORY}"
aws ecr describe-repositories --region "${AWS_REGION}" --repository-names "${REPOSITORY}" \
    >/dev/null 2>&1 ||
    aws ecr create-repository --region "${AWS_REGION}" --repository-name "${REPOSITORY}" \
        --image-scanning-configuration scanOnPush=true \
        --encryption-configuration encryptionType=AES256 >/dev/null

if [ -n "${SKIP_IMAGE_BUILD:-}" ]; then
    echo "==> 2/4 Build and push the job image ${IMAGE_URI} (SKIP_IMAGE_BUILD set)"
    echo "    Run these on your Docker-capable host, from the repository root, with the aws"
    echo "    CLI and deployer credentials reachable there (env vars or ~/.aws - no Python"
    echo "    required for either):"
    echo
    echo "    aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${REGISTRY}"
    echo "    docker build -f infrastructure/docker/Dockerfile -t ${IMAGE_URI} ."
    echo "    docker push ${IMAGE_URI}"
else
    echo "==> 2/4 Build and push the job image ${IMAGE_URI}"
    # This is the image SageMaker runs. It is not the control-plane image that compose builds.
    aws ecr get-login-password --region "${AWS_REGION}" |
        docker login --username AWS --password-stdin "${REGISTRY}"
    docker build \
        --build-arg "http_proxy=${http_proxy:-}" \
        --build-arg "https_proxy=${https_proxy:-}" \
        -f infrastructure/docker/Dockerfile \
        -t "${IMAGE_URI}" .
    docker push "${IMAGE_URI}"
fi

echo "==> 3/4 Upload the workflow definitions"
aws s3 cp infrastructure/stepfunctions/ "s3://${DEFINITIONS_BUCKET}/ml-factory/stepfunctions/" \
    --region "${AWS_REGION}" --recursive --exclude "*" --include "*.asl.json"

echo "==> 4/4 Deploy the stack ${STACK_NAME}"
aws cloudformation deploy \
    --region "${AWS_REGION}" \
    --template-file infrastructure/cloudformation/ml-factory.yaml \
    --stack-name "${STACK_NAME}" \
    --capabilities CAPABILITY_NAMED_IAM \
    --parameter-overrides \
    "ArtifactBucketName=${ARTIFACT_BUCKET}" \
    "ApprovedDataBucketName=${APPROVED_DATA_BUCKET}" \
    "ApprovedDataPrefix=${APPROVED_DATA_PREFIX}" \
    "JobImageUri=${IMAGE_URI}" \
    "DefinitionsBucket=${DEFINITIONS_BUCKET}" \
    "ExistingBackendRoleArn=${EXISTING_BACKEND_ROLE_ARN:-}" \
    "ExistingWorkflowRoleArn=${EXISTING_WORKFLOW_ROLE_ARN:-}" \
    "ExistingJobRoleArn=${EXISTING_JOB_ROLE_ARN:-}"

echo
echo "Done. Copy these into config/secrets/config.py:"
aws cloudformation describe-stacks --region "${AWS_REGION}" --stack-name "${STACK_NAME}" \
    --query 'Stacks[0].Outputs[].[OutputKey,OutputValue]' --output table
