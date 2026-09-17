#!/usr/bin/env python3
"""One-time (and on-upgrade) AWS deployment for the ML Factory — no AWS CLI required.

Does exactly what scripts/deploy_aws.sh does, over boto3 instead of shelling out to the `aws`
binary. Docker is still required: SageMaker runs a container image pulled from ECR, and there
is no SDK call that replaces building and pushing one. Everything else — the ECR repository,
the workflow upload, and the CloudFormation stack (S3 + KMS + IAM + the two state machines) —
goes through boto3 directly.

    python scripts/deploy_aws.py

    # If a create/update fails with a bare "Validation failed with N error(s)" and no other
    # detail (CloudFormation's own template/schema validation, rejected before any resource is
    # touched — DescribeStackEvents never carries the itemized errors for this failure mode):
    python scripts/deploy_aws.py --diagnose

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
    SKIP_IMAGE_BUILD ("1"/"true")   for when Docker and this script run on two different
    machines. Skips docker_login/build/push and instead prints the exact commands to run on
    the Docker-capable host, then continues on to the workflow upload and the CloudFormation
    deploy — CloudFormation does not check that the image already exists in ECR.

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
import contextlib
import importlib.util
import json
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
    "SKIP_IMAGE_BUILD",
)

_TRUTHY = {"1", "true", "yes", "on"}


def _flag(name: str) -> bool:
    return (_resolve(name) or "").strip().lower() in _TRUTHY


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


def preflight(session, parameters: dict[str, str]) -> None:
    """Check what CloudFormation's Early Validation checks, but name the setting that is wrong.

    A stack referencing a bucket or a role that does not exist is rejected with one opaque
    "Validation failed with 1 error(s)" and no indication of which reference it objected to.
    Checking first turns that post-mortem into the name of a value to fix, and does it before
    an image build rather than after one.
    """
    from botocore.exceptions import ClientError

    print("==> 0/4 Checking the resources the stack expects to already exist")
    s3 = session.client("s3")
    iam = session.client("iam")
    problems: list[str] = []

    for setting, key in (
        ("APPROVED_DATA_BUCKET", "ApprovedDataBucketName"),
        ("DEFINITIONS_BUCKET", "DefinitionsBucket"),
    ):
        bucket = parameters[key]
        try:
            s3.head_bucket(Bucket=bucket)
            print(f"    {setting} s3://{bucket} ok")
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code", "")
            problems.append(f"{setting}={bucket!r}: {'no such bucket' if code == '404' else error}")

    for setting, key in (
        ("EXISTING_BACKEND_ROLE_ARN", "ExistingBackendRoleArn"),
        ("EXISTING_WORKFLOW_ROLE_ARN", "ExistingWorkflowRoleArn"),
        ("EXISTING_JOB_ROLE_ARN", "ExistingJobRoleArn"),
    ):
        arn = parameters.get(key) or ""
        if not arn:
            # Empty is the default and means the stack creates that role itself.
            continue
        try:
            iam.get_role(RoleName=arn.rpartition("/")[2])
            print(f"    {setting} {arn} ok")
        except ClientError as error:
            problems.append(f"{setting}={arn!r}: {error}")

    if problems:
        print("\nerror: fix these in config/secrets/deploy.py before deploying:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        sys.exit(1)


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


def print_manual_image_commands(registry: str, image_uri: str, *, region: str) -> None:
    """For SKIP_IMAGE_BUILD: when Docker only exists on a different host than this script runs
    on, there is no session or pipe to hand it — only text to copy into that other terminal.
    This never decodes or prints a credential itself; it prints the same `aws ecr
    get-login-password | docker login` pipeline the CLI path already uses, so the token is
    fetched and consumed on the Docker host, in the same process, and never appears as text.
    That host needs the `aws` CLI and deployer credentials reachable there (env vars or a
    mounted `~/.aws` profile — no Python required for either).
    """
    print(f"==> 2/4 Build and push the job image {image_uri} (SKIP_IMAGE_BUILD set)")
    print("    Run these on your Docker-capable host, from the repository root:")
    print()
    print(
        f"    aws ecr get-login-password --region {region} | "
        f"docker login --username AWS --password-stdin {registry}"
    )
    print(f"    docker build -f infrastructure/docker/Dockerfile -t {image_uri} .")
    print(f"    docker push {image_uri}")


def upload_workflow_definitions(s3, bucket: str) -> None:
    """Both state machines reference these objects by exact key, so uploading nothing leaves
    the stack pointing at objects that do not exist — which CloudFormation rejects with an
    opaque validation error rather than a missing-object one. Say what was uploaded, and stop
    here rather than deploy a stack that cannot work."""
    print("==> 3/4 Upload the workflow definitions")
    source = ROOT / "infrastructure" / "stepfunctions"
    paths = sorted(source.glob("*.asl.json"))
    if not paths:
        print(f"error: no *.asl.json definitions found under {source}", file=sys.stderr)
        sys.exit(1)
    for path in paths:
        key = f"ml-factory/stepfunctions/{path.name}"
        s3.upload_file(str(path), bucket, key)
        print(f"    s3://{bucket}/{key}")


def _stack_status(cfn, stack_name: str) -> str | None:
    """None if the stack does not exist yet, otherwise its current StackStatus."""
    try:
        return cfn.describe_stacks(StackName=stack_name)["Stacks"][0]["StackStatus"]
    except cfn.exceptions.ClientError as error:
        if "does not exist" not in str(error):
            raise
        return None


def _print_stack_failure_reasons(cfn, stack_name: str) -> None:
    """CloudFormation's own waiters only report a status, not why — the failing events carry
    the actual reason (an IAM error, a bucket that already exists, a bad parameter, ...).

    Two different kinds of event have to be read for that. A resource failure reports through
    ResourceStatus/ResourceStatusReason. A *hook* failure — an Early Validation check, say,
    which is what rejects a template referencing a resource that does not exist — reports
    through HookStatus/HookStatusReason instead and leaves ResourceStatus absent entirely, so
    reading only the first kind makes exactly the failure that needs explaining invisible.
    """
    print(f"    {stack_name} failed. The failing resources:")
    events = cfn.describe_stack_events(StackName=stack_name)["StackEvents"]
    for event in reversed(events):
        if event.get("ResourceStatus", "").endswith("_FAILED"):
            reason = event.get("ResourceStatusReason", "")
            print(f"      {event['LogicalResourceId']} ({event['ResourceStatus']}): {reason}")
        if event.get("HookStatus", "").endswith("_FAILED"):
            hook = event.get("HookType", "hook")
            reason = event.get("HookStatusReason", "")
            print(f"      {event['LogicalResourceId']} [{hook}]: {reason}")


def _print_parameters(parameters: dict[str, str]) -> None:
    """What the stack is actually being given. An empty Existing*RoleArn means the stack
    creates that role itself, which is the default and not a missing value — printing them
    makes the difference visible rather than something to infer from a config file."""
    width = max((len(name) for name in parameters), default=0)
    for name, value in parameters.items():
        print(f"    {name:<{width}}  {value or '(empty)'}")


def deploy_stack(cfn, *, stack_name: str, parameters: dict[str, str]) -> None:
    from botocore.exceptions import WaiterError

    print(f"==> 4/4 Deploy the stack {stack_name}")
    _print_parameters(parameters)
    template_body = (ROOT / "infrastructure" / "cloudformation" / "ml-factory.yaml").read_text()
    cfn_parameters = [{"ParameterKey": k, "ParameterValue": v} for k, v in parameters.items()]

    status = _stack_status(cfn, stack_name)
    if status == "ROLLBACK_COMPLETE":
        # A stack stuck here is from a create that failed and rolled back; CloudFormation
        # refuses to update it — the only way forward is to delete it and create it again.
        print(
            f"    {stack_name} is in ROLLBACK_COMPLETE from an earlier failed create;"
            " deleting it before retrying"
        )
        cfn.delete_stack(StackName=stack_name)
        cfn.get_waiter("stack_delete_complete").wait(
            StackName=stack_name, WaiterConfig={"Delay": 10, "MaxAttempts": 180}
        )
        status = None

    common = {
        "StackName": stack_name,
        "TemplateBody": template_body,
        "Parameters": cfn_parameters,
        "Capabilities": ["CAPABILITY_NAMED_IAM"],
    }
    if status is not None:
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
    try:
        waiter.wait(StackName=stack_name, WaiterConfig={"Delay": 10, "MaxAttempts": 180})
    except WaiterError:
        _print_stack_failure_reasons(cfn, stack_name)
        raise


def print_outputs(cfn, stack_name: str) -> None:
    print("\nDone. Copy these into config/secrets/config.py:")
    outputs = cfn.describe_stacks(StackName=stack_name)["Stacks"][0].get("Outputs", [])
    width = max((len(o["OutputKey"]) for o in outputs), default=0)
    for output in outputs:
        print(f"  {output['OutputKey']:<{width}}  {output['OutputValue']}")


def _deploy_inputs() -> tuple[str, str, str, dict[str, str]]:
    """The settings shared by main() and --diagnose: (region, stack_name, image_uri,
    stack_parameters), read the same way every time."""
    region = env("AWS_REGION", required=True)
    account_id = env("AWS_ACCOUNT_ID", required=True)
    artifact_bucket = env("ARTIFACT_BUCKET", required=True)
    approved_data_bucket = env("APPROVED_DATA_BUCKET", required=True)
    definitions_bucket = env("DEFINITIONS_BUCKET", required=True)
    approved_data_prefix = env("APPROVED_DATA_PREFIX", "curated")
    stack_name = env("STACK_NAME", "ml-factory")
    image_tag = env("IMAGE_TAG") or git_short_sha()

    registry = f"{account_id}.dkr.ecr.{region}.amazonaws.com"
    image_uri = f"{registry}/ml-factory-jobs:{image_tag}"

    parameters = {
        "ArtifactBucketName": artifact_bucket,
        "ApprovedDataBucketName": approved_data_bucket,
        "ApprovedDataPrefix": approved_data_prefix,
        "JobImageUri": image_uri,
        "DefinitionsBucket": definitions_bucket,
        "ExistingBackendRoleArn": env("EXISTING_BACKEND_ROLE_ARN"),
        "ExistingWorkflowRoleArn": env("EXISTING_WORKFLOW_ROLE_ARN"),
        "ExistingJobRoleArn": env("EXISTING_JOB_ROLE_ARN"),
    }
    return region, stack_name, image_uri, parameters


def diagnose_stack(cfn, *, stack_name: str, parameters: dict[str, str]) -> None:
    """For a CREATE_FAILED stack whose only failing "resource" is the stack itself, reason
    "Validation failed with N error(s)...": that is CloudFormation's own schema validation
    rejecting the template before it touches a single resource, and DescribeStackEvents never
    carries the itemized detail for this failure mode — only a change set's StatusReason does.
    Runs against a disposable, differently-named stack so the real one is never touched.
    """
    from botocore.exceptions import WaiterError

    diagnostic_name = f"{stack_name}-diagnose"
    template_body = (ROOT / "infrastructure" / "cloudformation" / "ml-factory.yaml").read_text()
    cfn_parameters = [{"ParameterKey": k, "ParameterValue": v} for k, v in parameters.items()]

    print(f"==> Creating a disposable change set ({diagnostic_name}) to see the full errors")
    _print_parameters(parameters)
    created = cfn.create_change_set(
        StackName=diagnostic_name,
        TemplateBody=template_body,
        Parameters=cfn_parameters,
        Capabilities=["CAPABILITY_NAMED_IAM"],
        ChangeSetType="CREATE",
        ChangeSetName="diagnose",
    )
    with contextlib.suppress(WaiterError):
        cfn.get_waiter("change_set_create_complete").wait(ChangeSetName=created["Id"])

    detail = cfn.describe_change_set(ChangeSetName=created["Id"])
    print(f"    Status: {detail.get('Status')}")
    print(f"    StatusReason: {detail.get('StatusReason', '(none)')}")

    # The StatusReason above names only the hook. Which resource it objected to is in this
    # operation's own events — read them before the cleanup below takes the change set away.
    print_operation_events(cfn, ChangeSetName=created["Id"])
    with contextlib.suppress(cfn.exceptions.ClientError):
        _print_stack_failure_reasons(cfn, diagnostic_name)

    cfn.delete_change_set(ChangeSetName=created["Id"])
    # A CREATE-type change set creates the stack (in REVIEW_IN_PROGRESS) even when the change
    # set itself fails validation; clean it up so nothing disposable is left behind.
    with contextlib.suppress(cfn.exceptions.ClientError):
        cfn.delete_stack(StackName=diagnostic_name)


def print_operation_events(cfn, **target: str) -> None:
    """DescribeEvents — a different API from DescribeStackEvents, and the one CloudFormation's
    own "Call DescribeEvents to retrieve the full list of issues" message means literally.

    Only its OperationEvents carry the Validation* fields (ValidationName, ValidationPath,
    ValidationStatusReason), which is where an Early Validation check says which resource it
    objected to. DescribeStackEvents has no such fields at all, so that failure is unreadable
    there no matter which of its keys you look under.

    ``target`` is StackName=... or ChangeSetName=... — a change set that failed validation is
    the more direct target of the two, being a single operation with nothing else in it.
    """
    label = next(iter(target.values()), "")
    print(f"==> Operation events for {label} (DescribeEvents)")
    describe = getattr(cfn, "describe_events", None)
    if describe is None:
        print("    (this boto3 is too old for DescribeEvents; pip install -U boto3)")
        return
    try:
        events = describe(**target, Filters={"FailedEvents": True}).get("OperationEvents", [])
        if not events:
            # Not every validation failure is flagged as a "failed event"; take the whole
            # operation rather than report nothing.
            events = describe(**target).get("OperationEvents", [])
    except cfn.exceptions.ClientError as error:
        print(f"    (unavailable: {error})")
        return
    if not events:
        print("    (none)")
    for event in events:
        print(json.dumps(event, indent=2, default=str, sort_keys=True))


def print_raw_stack_events(cfn, stack_name: str, *, limit: int = 25) -> None:
    """Every field of the most recent events, verbatim.

    A failure that reports through fields this script does not anticipate — an Early Validation
    hook was the case that forced this — is invisible to any reporter that filters on the
    fields it happens to know about. So print what CloudFormation actually returned and read it,
    rather than guessing which key the answer is under.
    """
    print(f"==> Raw stack events for {stack_name} (most recent {limit})")
    try:
        events = cfn.describe_stack_events(StackName=stack_name)["StackEvents"][:limit]
    except cfn.exceptions.ClientError as error:
        print(f"    (no events: {error})")
        return
    for event in events:
        print(json.dumps(event, indent=2, default=str, sort_keys=True))


def diagnose() -> None:
    _region, stack_name, _image_uri, parameters = _deploy_inputs()
    session = build_session()
    _print_parameters(parameters)
    preflight(session, parameters)
    cfn = session.client("cloudformation")
    # The real stack is the one that actually attempted a create, so it is the one whose events
    # carry the validation failure's detail; the change-set probe below never gets that far.
    print_operation_events(cfn, StackName=stack_name)
    print_raw_stack_events(cfn, stack_name)
    diagnose_stack(cfn, stack_name=stack_name, parameters=parameters)


def main() -> None:
    region, stack_name, image_uri, parameters = _deploy_inputs()
    account_id = env("AWS_ACCOUNT_ID", required=True)
    registry = f"{account_id}.dkr.ecr.{region}.amazonaws.com"
    repository = "ml-factory-jobs"

    session = build_session()
    ecr = session.client("ecr")
    s3 = session.client("s3")
    cfn = session.client("cloudformation")

    preflight(session, parameters)
    ensure_ecr_repository(ecr, repository)
    if _flag("SKIP_IMAGE_BUILD"):
        print_manual_image_commands(registry, image_uri, region=region)
    else:
        docker_login(ecr, registry)
        build_and_push_image(image_uri)
    upload_workflow_definitions(s3, env("DEFINITIONS_BUCKET", required=True))
    deploy_stack(cfn, stack_name=stack_name, parameters=parameters)
    print_outputs(cfn, stack_name)


if __name__ == "__main__":
    if "--print-shell-exports" in sys.argv:
        print_shell_exports()
    elif "--diagnose" in sys.argv:
        diagnose()
    else:
        main()
