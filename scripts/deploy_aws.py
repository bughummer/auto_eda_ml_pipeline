#!/usr/bin/env python3
"""One-time (and on-upgrade) AWS deployment for the ML Factory — no AWS CLI required.

Does exactly what scripts/deploy_aws.sh does, over boto3 instead of shelling out to the `aws`
binary. Docker is still required: SageMaker runs a container image pulled from ECR, and there
is no SDK call that replaces building and pushing one. Everything else — the ECR repository,
the workflow upload, and the CloudFormation stack (S3 + KMS + IAM + the two state machines) —
goes through boto3 directly.

    python scripts/deploy_aws.py

Required environment:
    AWS_REGION              e.g. eu-central-1
    AWS_ACCOUNT_ID          e.g. 123456789012
    ARTIFACT_BUCKET         bucket the stack creates for artifacts and experiment records
    APPROVED_DATA_BUCKET    existing bucket holding approved datasets
    DEFINITIONS_BUCKET      existing bucket the workflow definitions are uploaded to

Optional:
    APPROVED_DATA_PREFIX    default: curated
    IMAGE_TAG               default: the short git SHA
    STACK_NAME              default: ml-factory
    EXISTING_BACKEND_ROLE_ARN    use this role instead of creating MlFactoryBackendRole
    EXISTING_WORKFLOW_ROLE_ARN   use this role instead of creating MlFactoryWorkflowRole
    EXISTING_JOB_ROLE_ARN        use this role instead of creating MlFactoryJobRole
    Each Existing*RoleArn role must already carry the matching policy in infrastructure/iam/
    (backend_role_policy.json / workflow_role_policy.json / job_role_policy.json) — this
    script does not create or modify a role you supply.

Credentials, in order of preference — the same chain boto3 always uses, made explicit here so
a deployer who has no CLI configured knows exactly what to set:
    1. DEPLOYER_ROLE_ARN set -> whatever base identity is otherwise found (env vars, a
       profile, an attached role) calls sts:AssumeRole on it, and every AWS call below runs
       as that role instead. The base identity only needs sts:AssumeRole on this one ARN;
       the role itself needs infrastructure/iam/deployer_policy.json.
    2. AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (+ AWS_SESSION_TOKEN if they are temporary,
       e.g. already produced by someone else's assume-role call) -> used directly. This
       identity needs infrastructure/iam/deployer_policy.json itself.
    3. Nothing set -> boto3's default chain (~/.aws, an attached role, ...), exactly as every
       other AWS call in this repository resolves credentials.
No credential of any shape is read from a file this script writes or logs.
"""

from __future__ import annotations

import base64
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def env(name: str, default: str | None = None, *, required: bool = False) -> str:
    value = os.environ.get(name, default)
    if required and not value:
        print(f"error: set {name}", file=sys.stderr)
        sys.exit(1)
    return value or ""


def git_short_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "latest"


def build_session():
    """The credential chain described in the module docstring, as boto3 calls."""
    import boto3

    region = env("AWS_REGION", required=True)
    role_arn = os.environ.get("DEPLOYER_ROLE_ARN")

    session_kwargs: dict[str, str] = {"region_name": region}
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if access_key and secret_key:
        session_kwargs["aws_access_key_id"] = access_key
        session_kwargs["aws_secret_access_key"] = secret_key
        if os.environ.get("AWS_SESSION_TOKEN"):
            session_kwargs["aws_session_token"] = os.environ["AWS_SESSION_TOKEN"]

    base_session = boto3.session.Session(**session_kwargs)
    if not role_arn:
        return base_session

    print(f"==> Assuming {role_arn}")
    sts = base_session.client("sts")
    assumed = sts.assume_role(RoleArn=role_arn, RoleSessionName="ml-factory-deploy")["Credentials"]
    return boto3.session.Session(
        region_name=region,
        aws_access_key_id=assumed["AccessKeyId"],
        aws_secret_access_key=assumed["SecretAccessKey"],
        aws_session_token=assumed["SessionToken"],
    )


def ensure_ecr_repository(ecr, repository: str) -> None:
    print(f"==> 1/4 ECR repository {repository}")
    try:
        ecr.describe_repositories(repositoryNames=[repository])
    except ecr.exceptions.RepositoryNotFoundException:
        ecr.create_repository(
            repositoryName=repository,
            imageScanningConfiguration={"scanOnPush": True},
            encryptionConfiguration={"encryptionType": "AES256"},
        )


def docker_login(ecr, registry: str) -> None:
    """The one thing ``aws ecr get-login-password`` does: mint a token and feed it to Docker.

    The token itself never touches disk or a log line — it is piped straight into the
    already-running ``docker login`` process, the same way the CLI pipeline does it.
    """
    auth = ecr.get_authorization_token()["authorizationData"][0]
    username, password = base64.b64decode(auth["authorizationToken"]).decode().split(":", 1)
    subprocess.run(
        ["docker", "login", "--username", username, "--password-stdin", registry],
        input=password,
        text=True,
        check=True,
    )


def build_and_push_image(image_uri: str) -> None:
    print(f"==> 2/4 Build and push the job image {image_uri}")
    # This is the image SageMaker runs. It is not the control-plane image that compose builds.
    subprocess.run(
        [
            "docker",
            "build",
            "--build-arg",
            f"http_proxy={os.environ.get('http_proxy', '')}",
            "--build-arg",
            f"https_proxy={os.environ.get('https_proxy', '')}",
            "-f",
            "infrastructure/docker/Dockerfile",
            "-t",
            image_uri,
            ".",
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(["docker", "push", image_uri], check=True)


def upload_workflow_definitions(s3, bucket: str) -> None:
    print("==> 3/4 Upload the workflow definitions")
    source = ROOT / "infrastructure" / "stepfunctions"
    for path in sorted(source.glob("*.asl.json")):
        key = f"ml-factory/stepfunctions/{path.name}"
        s3.upload_file(str(path), bucket, key)


def deploy_stack(cfn, *, stack_name: str, parameters: dict[str, str]) -> None:
    print(f"==> 4/4 Deploy the stack {stack_name}")
    template_body = (ROOT / "infrastructure" / "cloudformation" / "ml-factory.yaml").read_text()
    cfn_parameters = [{"ParameterKey": k, "ParameterValue": v} for k, v in parameters.items()]

    exists = True
    try:
        cfn.describe_stacks(StackName=stack_name)
    except cfn.exceptions.ClientError as error:
        if "does not exist" not in str(error):
            raise
        exists = False

    common = {
        "StackName": stack_name,
        "TemplateBody": template_body,
        "Parameters": cfn_parameters,
        "Capabilities": ["CAPABILITY_NAMED_IAM"],
    }
    if exists:
        try:
            cfn.update_stack(**common)
        except cfn.exceptions.ClientError as error:
            if "No updates are to be performed" in str(error):
                print("    no changes")
                return
            raise
        waiter = cfn.get_waiter("stack_update_complete")
    else:
        cfn.create_stack(**common, OnFailure="ROLLBACK")
        waiter = cfn.get_waiter("stack_create_complete")

    print("    waiting for CloudFormation...")
    waiter.wait(StackName=stack_name, WaiterConfig={"Delay": 10, "MaxAttempts": 180})


def print_outputs(cfn, stack_name: str) -> None:
    print("\nDone. Copy these into config/secrets/config.py:")
    outputs = cfn.describe_stacks(StackName=stack_name)["Stacks"][0].get("Outputs", [])
    width = max((len(o["OutputKey"]) for o in outputs), default=0)
    for output in outputs:
        print(f"  {output['OutputKey']:<{width}}  {output['OutputValue']}")


def main() -> None:
    region = env("AWS_REGION", required=True)
    account_id = env("AWS_ACCOUNT_ID", required=True)
    artifact_bucket = env("ARTIFACT_BUCKET", required=True)
    approved_data_bucket = env("APPROVED_DATA_BUCKET", required=True)
    definitions_bucket = env("DEFINITIONS_BUCKET", required=True)
    approved_data_prefix = env("APPROVED_DATA_PREFIX", "curated")
    stack_name = env("STACK_NAME", "ml-factory")
    image_tag = env("IMAGE_TAG") or git_short_sha()
    existing_backend_role_arn = env("EXISTING_BACKEND_ROLE_ARN")
    existing_workflow_role_arn = env("EXISTING_WORKFLOW_ROLE_ARN")
    existing_job_role_arn = env("EXISTING_JOB_ROLE_ARN")

    repository = "ml-factory-jobs"
    registry = f"{account_id}.dkr.ecr.{region}.amazonaws.com"
    image_uri = f"{registry}/{repository}:{image_tag}"

    session = build_session()
    ecr = session.client("ecr")
    s3 = session.client("s3")
    cfn = session.client("cloudformation")

    ensure_ecr_repository(ecr, repository)
    docker_login(ecr, registry)
    build_and_push_image(image_uri)
    upload_workflow_definitions(s3, definitions_bucket)
    deploy_stack(
        cfn,
        stack_name=stack_name,
        parameters={
            "ArtifactBucketName": artifact_bucket,
            "ApprovedDataBucketName": approved_data_bucket,
            "ApprovedDataPrefix": approved_data_prefix,
            "JobImageUri": image_uri,
            "DefinitionsBucket": definitions_bucket,
            "ExistingBackendRoleArn": existing_backend_role_arn,
            "ExistingWorkflowRoleArn": existing_workflow_role_arn,
            "ExistingJobRoleArn": existing_job_role_arn,
        },
    )
    print_outputs(cfn, stack_name)


if __name__ == "__main__":
    main()
