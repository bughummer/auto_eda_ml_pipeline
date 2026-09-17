"""scripts/deploy_aws.py — the boto3 deploy path, with every AWS/Docker call stubbed.

This exercises the branching logic that has no CLI equivalent to fall back on: create vs.
update, swallowing CloudFormation's "no changes" error, the assume-role credential chain, and
that the parameters sent to the stack match what the template actually declares. None of this
touches a network.
"""

import base64
import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
deploy_aws = importlib.import_module("deploy_aws")

REQUIRED_ENV = {
    "AWS_REGION": "eu-central-1",
    "AWS_ACCOUNT_ID": "123456789012",
    "ARTIFACT_BUCKET": "my-artifacts",
    "APPROVED_DATA_BUCKET": "my-data",
    "DEFINITIONS_BUCKET": "my-definitions",
    "IMAGE_TAG": "test-tag",
}


@pytest.fixture(autouse=True)
def required_environment(monkeypatch):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    for stray in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "DEPLOYER_ROLE_ARN",
        "EXISTING_BACKEND_ROLE_ARN",
        "EXISTING_WORKFLOW_ROLE_ARN",
        "EXISTING_JOB_ROLE_ARN",
    ):
        monkeypatch.delenv(stray, raising=False)


class FakeClientError(Exception):
    pass


def test_the_cloudformation_parameters_match_the_template_exactly():
    """A parameter name here that the template does not declare fails the whole deploy."""
    template = (
        Path(__file__).resolve().parents[2]
        / "infrastructure"
        / "cloudformation"
        / "ml-factory.yaml"
    ).read_text()

    cfn = MagicMock()
    cfn.exceptions.ClientError = FakeClientError
    cfn.describe_stacks.side_effect = FakeClientError("Stack ml-factory does not exist")

    sent_parameters = {}

    def capture_create(**kwargs):
        for entry in kwargs["Parameters"]:
            sent_parameters[entry["ParameterKey"]] = entry["ParameterValue"]

    cfn.create_stack.side_effect = capture_create
    cfn.get_waiter.return_value = MagicMock()

    deploy_aws.deploy_stack(
        cfn,
        stack_name="ml-factory",
        parameters={
            "ArtifactBucketName": "b",
            "ApprovedDataBucketName": "d",
            "ApprovedDataPrefix": "curated",
            "JobImageUri": "uri",
            "DefinitionsBucket": "f",
        },
    )

    for key in sent_parameters:
        assert f"{key}:" in template, f"{key} is not a Parameter in ml-factory.yaml"
    assert set(sent_parameters) == {
        "ArtifactBucketName",
        "ApprovedDataBucketName",
        "ApprovedDataPrefix",
        "JobImageUri",
        "DefinitionsBucket",
    }


def test_a_missing_stack_is_created_not_updated():
    cfn = MagicMock()
    cfn.exceptions.ClientError = FakeClientError
    cfn.describe_stacks.side_effect = FakeClientError("Stack ml-factory does not exist")

    deploy_aws.deploy_stack(cfn, stack_name="ml-factory", parameters={"ArtifactBucketName": "b"})

    cfn.create_stack.assert_called_once()
    cfn.update_stack.assert_not_called()
    assert cfn.create_stack.call_args.kwargs["OnFailure"] == "ROLLBACK"
    cfn.get_waiter.assert_called_with("stack_create_complete")


def test_an_existing_stack_is_updated_not_recreated():
    cfn = MagicMock()
    cfn.exceptions.ClientError = FakeClientError
    cfn.describe_stacks.return_value = {
        "Stacks": [{"StackName": "ml-factory", "StackStatus": "CREATE_COMPLETE"}]
    }

    deploy_aws.deploy_stack(cfn, stack_name="ml-factory", parameters={"ArtifactBucketName": "b"})

    cfn.update_stack.assert_called_once()
    cfn.create_stack.assert_not_called()
    cfn.delete_stack.assert_not_called()
    cfn.get_waiter.assert_called_with("stack_update_complete")


def test_no_pending_changes_is_success_not_an_error():
    """update_stack raises for 'no changes' where the CLI's `deploy` swallows it. Match that."""
    cfn = MagicMock()
    cfn.exceptions.ClientError = FakeClientError
    cfn.describe_stacks.return_value = {
        "Stacks": [{"StackName": "ml-factory", "StackStatus": "CREATE_COMPLETE"}]
    }
    cfn.update_stack.side_effect = FakeClientError(
        "An error occurred (ValidationError): No updates are to be performed."
    )

    deploy_aws.deploy_stack(cfn, stack_name="ml-factory", parameters={"ArtifactBucketName": "b"})

    cfn.get_waiter.assert_not_called()


def test_a_real_update_failure_is_not_swallowed():
    cfn = MagicMock()
    cfn.exceptions.ClientError = FakeClientError
    cfn.describe_stacks.return_value = {
        "Stacks": [{"StackName": "ml-factory", "StackStatus": "CREATE_COMPLETE"}]
    }
    cfn.update_stack.side_effect = FakeClientError("AccessDenied")

    with pytest.raises(FakeClientError, match="AccessDenied"):
        deploy_aws.deploy_stack(
            cfn, stack_name="ml-factory", parameters={"ArtifactBucketName": "b"}
        )


def test_a_stack_stuck_in_rollback_complete_is_deleted_then_recreated():
    """CloudFormation refuses to update a ROLLBACK_COMPLETE stack; the only way forward is to
    delete it and create it again."""
    cfn = MagicMock()
    cfn.exceptions.ClientError = FakeClientError
    cfn.describe_stacks.return_value = {
        "Stacks": [{"StackName": "ml-factory", "StackStatus": "ROLLBACK_COMPLETE"}]
    }

    deploy_aws.deploy_stack(cfn, stack_name="ml-factory", parameters={"ArtifactBucketName": "b"})

    cfn.delete_stack.assert_called_once_with(StackName="ml-factory")
    cfn.get_waiter.assert_any_call("stack_delete_complete")
    cfn.create_stack.assert_called_once()
    cfn.update_stack.assert_not_called()
    cfn.get_waiter.assert_any_call("stack_create_complete")


def test_a_failed_deploy_prints_the_actual_failing_resources_before_raising():
    """A raw WaiterError only says 'ROLLBACK_COMPLETE'; the *_FAILED stack events carry the
    reason (bad IAM trust policy, a bucket that already exists, ...) that a deployer needs."""
    from botocore.exceptions import WaiterError

    cfn = MagicMock()
    cfn.exceptions.ClientError = FakeClientError
    cfn.describe_stacks.side_effect = FakeClientError("Stack ml-factory does not exist")
    cfn.get_waiter.return_value.wait.side_effect = WaiterError(
        name="StackCreateComplete",
        reason="Waiter encountered a terminal failure state",
        last_response={},
    )
    cfn.describe_stack_events.return_value = {
        "StackEvents": [
            {
                "LogicalResourceId": "ArtifactBucket",
                "ResourceStatus": "CREATE_FAILED",
                "ResourceStatusReason": "bucket-name already exists",
            },
            {
                "LogicalResourceId": "ml-factory",
                "ResourceStatus": "ROLLBACK_IN_PROGRESS",
                "ResourceStatusReason": "",
            },
        ]
    }

    with pytest.raises(WaiterError):
        deploy_aws.deploy_stack(
            cfn, stack_name="ml-factory", parameters={"ArtifactBucketName": "b"}
        )

    cfn.describe_stack_events.assert_called_once_with(StackName="ml-factory")


def test_a_repository_that_already_exists_is_not_recreated():
    ecr = MagicMock()
    ecr.exceptions.RepositoryNotFoundException = FakeClientError
    ecr.describe_repositories.return_value = {}

    deploy_aws.ensure_ecr_repository(ecr, "ml-factory-jobs")

    ecr.create_repository.assert_not_called()


def test_a_missing_repository_is_created():
    ecr = MagicMock()
    ecr.exceptions.RepositoryNotFoundException = FakeClientError
    ecr.describe_repositories.side_effect = FakeClientError("not found")

    deploy_aws.ensure_ecr_repository(ecr, "ml-factory-jobs")

    ecr.create_repository.assert_called_once()
    kwargs = ecr.create_repository.call_args.kwargs
    assert kwargs["repositoryName"] == "ml-factory-jobs"
    assert kwargs["imageScanningConfiguration"] == {"scanOnPush": True}


def test_the_ecr_token_is_decoded_and_piped_to_docker_login_not_logged():
    ecr = MagicMock()
    token = base64.b64encode(b"AWS:supersecretpassword").decode()
    ecr.get_authorization_token.return_value = {
        "authorizationData": [{"authorizationToken": token}]
    }

    with patch("deploy_aws.subprocess.run") as run:
        deploy_aws.docker_login(ecr, "123456789012.dkr.ecr.eu-central-1.amazonaws.com")

    run.assert_called_once()
    args, kwargs = run.call_args
    assert args[0] == [
        "docker",
        "login",
        "--username",
        "AWS",
        "--password-stdin",
        "123456789012.dkr.ecr.eu-central-1.amazonaws.com",
    ]
    assert kwargs["input"] == "supersecretpassword"
    assert kwargs["check"] is True


def test_workflow_definitions_are_uploaded_under_the_fixed_prefix():
    s3 = MagicMock()

    deploy_aws.upload_workflow_definitions(s3, "my-definitions")

    uploaded_keys = {call_args.args[2] for call_args in s3.upload_file.call_args_list}
    assert uploaded_keys == {
        "ml-factory/stepfunctions/eda_state_machine.asl.json",
        "ml-factory/stepfunctions/training_state_machine.asl.json",
    }
    for call_args in s3.upload_file.call_args_list:
        assert call_args.args[1] == "my-definitions"


# --- credentials ------------------------------------------------------------------


def test_static_credentials_are_used_directly_when_no_role_is_configured(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")

    fake_boto3 = MagicMock()
    with patch.dict(sys.modules, {"boto3": fake_boto3}):
        deploy_aws.build_session()

    fake_boto3.session.Session.assert_called_once_with(
        region_name="eu-central-1", aws_access_key_id="AKIAEXAMPLE", aws_secret_access_key="secret"
    )
    fake_boto3.session.Session.return_value.client.assert_not_called()


def test_a_session_token_is_forwarded_when_the_static_credentials_are_temporary(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "session-token")

    fake_boto3 = MagicMock()
    with patch.dict(sys.modules, {"boto3": fake_boto3}):
        deploy_aws.build_session()

    kwargs = fake_boto3.session.Session.call_args.kwargs
    assert kwargs["aws_session_token"] == "session-token"


def test_a_deployer_role_arn_is_assumed_and_its_temporary_credentials_are_used(monkeypatch):
    """The base identity is used only to call AssumeRole; every later AWS call runs as the role."""
    monkeypatch.setenv("DEPLOYER_ROLE_ARN", "arn:aws:iam::123456789012:role/Deployer")

    fake_boto3 = MagicMock()
    base_session, assumed_session = MagicMock(), MagicMock()
    fake_boto3.session.Session.side_effect = [base_session, assumed_session]
    base_session.client.return_value.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "ASIAROLE",
            "SecretAccessKey": "role-secret",
            "SessionToken": "role-session-token",
        }
    }

    with patch.dict(sys.modules, {"boto3": fake_boto3}):
        result = deploy_aws.build_session()

    base_session.client.assert_called_once_with("sts")
    base_session.client.return_value.assume_role.assert_called_once_with(
        RoleArn="arn:aws:iam::123456789012:role/Deployer", RoleSessionName="ml-factory-deploy"
    )
    assert result is assumed_session
    second_call_kwargs = fake_boto3.session.Session.call_args_list[1].kwargs
    assert second_call_kwargs["aws_access_key_id"] == "ASIAROLE"
    assert second_call_kwargs["aws_secret_access_key"] == "role-secret"
    assert second_call_kwargs["aws_session_token"] == "role-session-token"


def test_a_required_variable_left_unset_fails_fast(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)

    with pytest.raises(SystemExit):
        deploy_aws.env("AWS_REGION", required=True)


# --- main(): the existing-role parameters actually reach CloudFormation ---------


def _run_main_with_everything_stubbed():
    """main() end to end, with the network and Docker replaced. Returns the create_stack call."""
    session = MagicMock()
    cfn = session.client.return_value
    cfn.exceptions.ClientError = FakeClientError
    # deploy_stack's existence check raises "does not exist" once; the create path then
    # succeeds, and print_outputs' own describe_stacks call afterward must not raise too.
    cfn.describe_stacks.side_effect = [
        FakeClientError("Stack ml-factory does not exist"),
        {"Stacks": [{"Outputs": []}]},
    ]
    cfn.get_authorization_token.return_value = {
        "authorizationData": [{"authorizationToken": base64.b64encode(b"AWS:pw").decode()}]
    }
    cfn.describe_repositories.return_value = {}

    with (
        patch("deploy_aws.build_session", return_value=session),
        patch("deploy_aws.subprocess.run"),
    ):
        deploy_aws.main()

    return cfn.create_stack.call_args.kwargs["Parameters"]


def test_by_default_no_existing_role_is_passed_so_the_template_creates_all_three():
    parameters = {
        p["ParameterKey"]: p["ParameterValue"] for p in _run_main_with_everything_stubbed()
    }

    assert parameters["ExistingBackendRoleArn"] == ""
    assert parameters["ExistingWorkflowRoleArn"] == ""
    assert parameters["ExistingJobRoleArn"] == ""


# --- SKIP_IMAGE_BUILD: Docker and this script on different machines -------------


def test_skip_image_build_prints_manual_commands_instead_of_running_docker():
    with patch("deploy_aws.subprocess.run") as run:
        deploy_aws.print_manual_image_commands(
            "123456789012.dkr.ecr.eu-central-1.amazonaws.com",
            "registry/repo:tag",
            region="eu-central-1",
        )

    run.assert_not_called()


def test_skip_image_build_manual_commands_never_decode_or_print_a_credential(capsys):
    """The pipeline fetches and consumes the ECR token on the Docker host in one step; this
    script must never hold or print a decoded password itself."""
    deploy_aws.print_manual_image_commands(
        "123456789012.dkr.ecr.eu-central-1.amazonaws.com",
        "registry/repo:tag",
        region="eu-central-1",
    )

    out = capsys.readouterr().out
    assert (
        "aws ecr get-login-password --region eu-central-1 | "
        "docker login --username AWS --password-stdin "
        "123456789012.dkr.ecr.eu-central-1.amazonaws.com" in out
    )
    assert "docker build -f infrastructure/docker/Dockerfile -t registry/repo:tag ." in out
    assert "docker push registry/repo:tag" in out


def test_skip_image_build_flag_recognises_common_truthy_spellings(monkeypatch):
    for value in ("1", "true", "True", "yes", "on"):
        monkeypatch.setenv("SKIP_IMAGE_BUILD", value)
        assert deploy_aws._flag("SKIP_IMAGE_BUILD") is True

    monkeypatch.setenv("SKIP_IMAGE_BUILD", "0")
    assert deploy_aws._flag("SKIP_IMAGE_BUILD") is False


def test_main_skips_docker_when_skip_image_build_is_set(monkeypatch):
    monkeypatch.setenv("SKIP_IMAGE_BUILD", "1")

    session = MagicMock()
    cfn = session.client.return_value
    cfn.exceptions.ClientError = FakeClientError
    cfn.describe_stacks.side_effect = [
        FakeClientError("Stack ml-factory does not exist"),
        {"Stacks": [{"Outputs": []}]},
    ]
    cfn.get_authorization_token.return_value = {
        "authorizationData": [{"authorizationToken": base64.b64encode(b"AWS:pw").decode()}]
    }
    cfn.describe_repositories.return_value = {}

    with (
        patch("deploy_aws.build_session", return_value=session),
        patch("deploy_aws.subprocess.run") as run,
    ):
        deploy_aws.main()

    run.assert_not_called()
    cfn.create_stack.assert_called_once()


def test_existing_role_arns_are_forwarded_to_the_stack(monkeypatch):
    monkeypatch.setenv("EXISTING_BACKEND_ROLE_ARN", "arn:aws:iam::123456789012:role/Backend")
    monkeypatch.setenv("EXISTING_WORKFLOW_ROLE_ARN", "arn:aws:iam::123456789012:role/Workflow")
    monkeypatch.setenv("EXISTING_JOB_ROLE_ARN", "arn:aws:iam::123456789012:role/Job")

    parameters = {
        p["ParameterKey"]: p["ParameterValue"] for p in _run_main_with_everything_stubbed()
    }

    assert parameters["ExistingBackendRoleArn"] == "arn:aws:iam::123456789012:role/Backend"
    assert parameters["ExistingWorkflowRoleArn"] == "arn:aws:iam::123456789012:role/Workflow"
    assert parameters["ExistingJobRoleArn"] == "arn:aws:iam::123456789012:role/Job"
