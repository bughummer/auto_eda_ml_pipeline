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
    cfn.describe_stacks.return_value = {"Stacks": [{"StackName": "ml-factory"}]}

    deploy_aws.deploy_stack(cfn, stack_name="ml-factory", parameters={"ArtifactBucketName": "b"})

    cfn.update_stack.assert_called_once()
    cfn.create_stack.assert_not_called()
    cfn.get_waiter.assert_called_with("stack_update_complete")


def test_no_pending_changes_is_success_not_an_error():
    """update_stack raises for 'no changes' where the CLI's `deploy` swallows it. Match that."""
    cfn = MagicMock()
    cfn.exceptions.ClientError = FakeClientError
    cfn.describe_stacks.return_value = {"Stacks": [{"StackName": "ml-factory"}]}
    cfn.update_stack.side_effect = FakeClientError(
        "An error occurred (ValidationError): No updates are to be performed."
    )

    deploy_aws.deploy_stack(cfn, stack_name="ml-factory", parameters={"ArtifactBucketName": "b"})

    cfn.get_waiter.assert_not_called()


def test_a_real_update_failure_is_not_swallowed():
    cfn = MagicMock()
    cfn.exceptions.ClientError = FakeClientError
    cfn.describe_stacks.return_value = {"Stacks": [{"StackName": "ml-factory"}]}
    cfn.update_stack.side_effect = FakeClientError("AccessDenied")

    with pytest.raises(FakeClientError, match="AccessDenied"):
        deploy_aws.deploy_stack(
            cfn, stack_name="ml-factory", parameters={"ArtifactBucketName": "b"}
        )


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
