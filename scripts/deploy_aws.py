#!/usr/bin/env python3
"""One-time (and on-upgrade) AWS deployment for the ML Factory — no AWS CLI required.

Does exactly what scripts/deploy_aws.sh does, over boto3 instead of shelling out to the `aws`
binary. Docker is still required: SageMaker runs a container image pulled from ECR, and there
is no SDK call that replaces building and pushing one. Everything else — the ECR repository,
the workflow upload, and the CloudFormation stack (S3 + KMS + IAM + the two state machines) —
goes through boto3 directly.

    python scripts/deploy_aws.py

Every setting below can be an environment variable, or a line in config/secrets/deploy.py
(copy config/secrets/deploy.sample.py) — the environment wins where both are set. Nothing
requires `export`; a filled-in deploy.py alone is enough to run this script.

Required:
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
    1. DEPLOYER_ROLE_ARN set -> whatever base identity is otherwise found (env vars,
       deploy.py, a profile, an attached role) calls sts:AssumeRole on it, and every AWS call
       below runs as that role instead. The base identity only needs sts:AssumeRole on this
       one ARN; the role itself needs infrastructure/iam/deployer_policy.json.
    2. AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (+ AWS_SESSION_TOKEN if they are temporary,
       e.g. already produced by someone else's assume-role call) -> used directly. This
       identity needs infrastructure/iam/deployer_policy.json itself.
    3. Nothing set -> boto3's default chain (~/.aws, an attached role, ...), exactly as every
       other AWS call in this repository resolves credentials.
config/secrets/deploy.py is gitignored, exactly like config/secrets/config.py; no credential
of any shape is otherwise written to a file or logged by this script.
"""

from __future__ import annotations

import base64
import importlib.util
import os
import shlex
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Where an operator types deploy-time settings instead of exporting them. Tracked sample:
#: config/secrets/deploy.sample.py. Overridable for tests, matching backend/config.py's
#: ML_FACTORY_SECRETS_FILE.
DEPLOY_SECRETS_FILE_ENV_VAR = "ML_FACTORY_DEPLOY_SECRETS_FILE"

#: Recognised by scripts/deploy_aws.sh's --print-shell-exports bridge (see below). deploy.py
#: itself may hold other names too — load_deploy_secrets() does not filter by this list, only
#: the shell bridge does, so it never exports something that was not meant as a setting.
_RECOGNISED_KEYS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "DEPLOYER_ROLE_ARN",
    "AWS_REGION",
    "AWS_ACCOUNT_ID",
    "ARTIFACT_BUCKET",
    "APPROVED_DATA_BUCKET",
    "DEFINITIONS_BUCKET",
    "APPROVED_DATA_PREFIX",
    "IMAGE_TAG",
    "STACK_NAME",
    "EXISTING_BACKEND_ROLE_ARN",
    "EXISTING_WORKFLOW_ROLE_ARN",
    "EXISTING_JOB_ROLE_ARN",
)


def _deploy_secrets_file() -> Path:
    return Path(
        os.environ.get(DEPLOY_SECRETS_FILE_ENV_VAR, str(ROOT / "config" / "secrets" / "deploy.py"))
    )


def load_deploy_secrets() -> dict[str, str]:
    """Values from config/secrets/deploy.py — an empty dict if the file does not exist.

    Loaded the same way backend/config.py reads config/secrets/config.py: module-level names
    are read directly, with imports, callables and names starting with ``_`` ignored, so the
    file can carry comments and helpers without them leaking in as settings. A value left as
    ``None`` (the sample's default for everything) is dropped rather than returned, so it
    falls through to the environment exactly as if the name were absent from the file.
    """
    path = _deploy_secrets_file()
    if not path.is_file():
        return {}
    spec = importlib.util.spec_from_file_location("ml_factory_deploy_secrets", path)
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {
        name: str(value)
        for name, value in vars(module).items()
        if not name.startswith("_")
        and value is not None
        and not isinstance(value, types.ModuleType)
        and not callable(value)
    }


def _resolve(name: str) -> str | None:
    """A setting's value: the environment first, config/secrets/deploy.py second."""
    return os.environ.get(name) or load_deploy_secrets().get(name)


def env(name: str, default: str | None = None, *, required: bool = False) -> str:
    value = _resolve(name) or default
    if required and not value:
        print(
            f"error: set {name} (as an environment variable, or in config/secrets/deploy.py)",
            file=sys.stderr,
        )
        sys.exit(1)
    return value or ""


def print_shell_exports() -> None:
    """Bridge for scripts/deploy_aws.sh: deploy.py's settings as `export KEY=value` lines.

    Only for the recognised keys, and only those the shell does not already have set — the
    same "environment wins" precedence env() applies, made visible to a process that is not
    this one. ``eval "$(python scripts/deploy_aws.py --print-shell-exports)"`` is how the bash
    script picks these up before it starts making its own `aws` CLI calls.
    """
    secrets = load_deploy_secrets()
    for key in _RECOGNISED_KEYS:
        if key in os.environ:
            continue
        value = secrets.get(key)
        if value:
            print(f"export {key}={shlex.quote(value)}")


def git_short_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "latest"


def credential_source() -> str:
    """How deploy credentials will be obtained. Safe to print — no values."""
    if _resolve("AWS_ACCESS_KEY_ID") and _resolve("AWS_SECRET_ACCESS_KEY"):
        base = "static access key"
    else:
        base = "default AWS chain (~/.aws, or an attached role)"
    role_arn = _resolve("DEPLOYER_ROLE_ARN")
    return f"{base}, assuming {role_arn}" if role_arn else base


def build_session():
    """The credential chain described in the module docstring, as boto3 calls."""
    import boto3

    print(f"==> Deploy credentials: {credential_source()}")
    region = env("AWS_REGION", required=True)
    role_arn = _resolve("DEPLOYER_ROLE_ARN")

    session_kwargs: dict[str, str] = {"region_name": region}
    access_key = _resolve("AWS_ACCESS_KEY_ID")
    secret_key = _resolve("AWS_SECRET_ACCESS_KEY")
    if access_key and secret_key:
        session_kwargs["aws_access_key_id"] = access_key
        session_kwargs["aws_secret_access_key"] = secret_key
        session_token = _resolve("AWS_SESSION_TOKEN")
        if session_token:
            session_kwargs["aws_session_token"] = session_token

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
    if "--print-shell-exports" in sys.argv:
        print_shell_exports()
    else:
        main()
