#!/usr/bin/env bash
# One-time (and on-upgrade) AWS deployment for the ML Factory.
#
# docker compose runs the control plane on this host. It cannot create AWS resources, and it
# cannot put the job image where SageMaker can pull it — SageMaker runs a container image from
# ECR, not a checkout. This script does both, in the order they have to happen.
#
#   ./scripts/deploy_aws.sh
#
# Required environment (or pass them inline):
#   AWS_REGION              e.g. eu-central-1
#   AWS_ACCOUNT_ID          e.g. 123456789012
#   ARTIFACT_BUCKET         bucket the stack creates for artifacts and experiment records
#   APPROVED_DATA_BUCKET    existing bucket holding approved datasets
#   DEFINITIONS_BUCKET      existing bucket the workflow definitions are uploaded to
# Optional:
#   APPROVED_DATA_PREFIX    default: curated
#   IMAGE_TAG               default: the short git SHA
#   STACK_NAME              default: ml-factory
set -euo pipefail

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
    "DefinitionsBucket=${DEFINITIONS_BUCKET}"

echo
echo "Done. Copy these into config/secrets/config.py:"
aws cloudformation describe-stacks --region "${AWS_REGION}" --stack-name "${STACK_NAME}" \
    --query 'Stacks[0].Outputs[].[OutputKey,OutputValue]' --output table
