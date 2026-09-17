"""infrastructure/cloudformation/ml-factory.yaml — the existing-role escape hatch.

Three runtime roles can each be supplied instead of created: an Existing*RoleArn parameter,
a Condition keyed off whether it was left empty, and every reference to that role's ARN
switched from a bare !GetAtt to an Fn::If between the created resource and the supplied ARN.
A regression here is silent at the YAML-syntax level — the template still parses — so these
assertions check the actual CloudFormation semantics: the right resources are conditional, the
right conditions gate them, and every consumer of a role's ARN honours the same condition.

cfn-lint provides the rest: real intrinsic-function validation, condition-combination coverage,
and drift on anything else in the template a future edit might disturb.
"""

from pathlib import Path

import pytest
import yaml

TEMPLATE = (
    Path(__file__).resolve().parents[2] / "infrastructure" / "cloudformation" / "ml-factory.yaml"
)


class _CfnLoader(yaml.SafeLoader):
    """Turns CloudFormation's short-form tags (!Ref, !GetAtt, !If, !Sub, ...) into the
    equivalent JSON intrinsic-function form, so the loaded document can be asserted on
    structurally instead of as opaque tagged scalars."""


def _construct_intrinsic(loader: yaml.SafeLoader, tag_suffix: str, node: yaml.Node):
    if tag_suffix == "GetAtt" and isinstance(node, yaml.ScalarNode):
        resource, _, attribute = loader.construct_scalar(node).partition(".")
        return {"Fn::GetAtt": [resource, attribute]}
    key = "Ref" if tag_suffix == "Ref" else f"Fn::{tag_suffix}"
    if isinstance(node, yaml.ScalarNode):
        return {key: loader.construct_scalar(node)}
    if isinstance(node, yaml.SequenceNode):
        return {key: loader.construct_sequence(node)}
    return {key: loader.construct_mapping(node)}


_CfnLoader.add_multi_constructor("!", _construct_intrinsic)


@pytest.fixture(scope="module")
def template() -> dict:
    return yaml.load(TEMPLATE.read_text(), Loader=_CfnLoader)


def _if_role_arn(condition: str, resource: str, existing_param: str) -> dict:
    """The Fn::If a role-ARN consumer must use: the created resource, or the supplied ARN."""
    return {"Fn::If": [condition, {"Fn::GetAtt": [resource, "Arn"]}, {"Ref": existing_param}]}


ROLES = [
    ("BackendRole", "CreateBackendRole", "ExistingBackendRoleArn"),
    ("WorkflowRole", "CreateWorkflowRole", "ExistingWorkflowRoleArn"),
    ("JobRole", "CreateJobRole", "ExistingJobRoleArn"),
]


@pytest.mark.parametrize(("resource", "condition", "param"), ROLES)
def test_each_existing_role_parameter_defaults_to_empty_so_the_role_is_created(
    template, resource, condition, param
):
    parameter = template["Parameters"][param]
    assert parameter["Type"] == "String"
    assert parameter["Default"] == "", "an empty default means 'create the role', not 'no role'"


@pytest.mark.parametrize(("resource", "condition", "param"), ROLES)
def test_each_condition_is_true_exactly_when_no_arn_was_supplied(
    template, resource, condition, param
):
    assert template["Conditions"][condition] == {"Fn::Equals": [{"Ref": param}, ""]}


@pytest.mark.parametrize(("resource", "condition", "param"), ROLES)
def test_each_role_resource_is_only_created_under_its_own_condition(
    template, resource, condition, param
):
    assert template["Resources"][resource]["Condition"] == condition


def test_the_workflow_role_assumes_the_created_or_supplied_backend_is_irrelevant_but_job_role_matters(
    template,
):
    """The one place a role ARN is consumed by another role's policy, not just a state machine."""
    statements = template["Resources"]["WorkflowRole"]["Properties"]["Policies"][0][
        "PolicyDocument"
    ]["Statement"]
    pass_role = next(s for s in statements if s["Action"] == ["iam:PassRole"])
    assert pass_role["Resource"] == _if_role_arn("CreateJobRole", "JobRole", "ExistingJobRoleArn")


@pytest.mark.parametrize("machine", ["EdaStateMachine", "TrainingStateMachine"])
def test_each_state_machine_runs_as_the_created_or_supplied_workflow_role(template, machine):
    properties = template["Resources"][machine]["Properties"]
    assert properties["RoleArn"] == _if_role_arn(
        "CreateWorkflowRole", "WorkflowRole", "ExistingWorkflowRoleArn"
    )


@pytest.mark.parametrize("machine", ["EdaStateMachine", "TrainingStateMachine"])
def test_each_state_machine_passes_the_created_or_supplied_job_role_to_sagemaker(template, machine):
    substitutions = template["Resources"][machine]["Properties"]["DefinitionSubstitutions"]
    assert substitutions["JobRoleArn"] == _if_role_arn(
        "CreateJobRole", "JobRole", "ExistingJobRoleArn"
    )


def test_the_backend_role_output_reflects_the_created_or_supplied_role(template):
    assert template["Outputs"]["BackendRoleArn"]["Value"] == _if_role_arn(
        "CreateBackendRole", "BackendRole", "ExistingBackendRoleArn"
    )


def test_a_role_created_by_this_template_is_never_referenced_without_its_condition(template):
    """Every !GetAtt on one of the three role resources must be wrapped in that role's Fn::If.

    A bare !GetAtt WorkflowRole.Arn left over from an incomplete edit would deploy fine when
    every role is created here, and fail with a confusing "resource does not exist" only when
    someone actually supplies an existing role — exactly the case this feature exists for.
    """
    owning_condition = {resource: condition for resource, condition, _ in ROLES}

    def walk(node, guard: str | None):
        if isinstance(node, dict):
            if "Fn::GetAtt" in node and len(node) == 1:
                resource, _attribute = node["Fn::GetAtt"]
                if resource in owning_condition:
                    assert guard == owning_condition[resource], (
                        f"Fn::GetAtt {resource} is reachable without {owning_condition[resource]}"
                        " guarding it - it must be the true-branch of that role's Fn::If"
                    )
                return
            if "Fn::If" in node:
                condition, if_true, if_false = node["Fn::If"]
                walk(if_true, condition)
                walk(if_false, guard)
                return
            for value in node.values():
                walk(value, guard)
        elif isinstance(node, list):
            for item in node:
                walk(item, guard)

    walk(template["Resources"], guard=None)
    walk(template["Outputs"], guard=None)


@pytest.mark.parametrize(
    ("reference", "resource"),
    [
        ("backend_role_policy.json", "BackendRole"),
        ("workflow_role_policy.json", "WorkflowRole"),
        ("job_role_policy.json", "JobRole"),
    ],
)
def test_each_role_grants_exactly_what_its_reference_policy_documents(
    template, reference, resource
):
    """infrastructure/iam/*.json is the reference version of each role, and the template holds
    the deployed equivalent — two copies of one intent, which drift silently. They have, twice:
    the workflow role lost the log-delivery actions Step Functions needs to create a state
    machine with logging at all, and the backend role lost bedrock:InvokeModel. Neither shows
    up until AWS refuses something at deploy or at runtime, so compare them here instead.
    """
    import json

    def actions(statements) -> set[str]:
        granted: set[str] = set()
        for statement in statements:
            # A conditional statement is an Fn::If whose true-branch is the statement itself.
            if "Fn::If" in statement:
                statement = statement["Fn::If"][1]
            action = statement["Action"]
            granted |= {action} if isinstance(action, str) else set(action)
        return granted

    reference_path = TEMPLATE.parents[1] / "iam" / reference
    documented = actions(json.loads(reference_path.read_text())["Statement"])
    policy = template["Resources"][resource]["Properties"]["Policies"][0]
    deployed = actions(policy["PolicyDocument"]["Statement"])

    assert deployed == documented, (
        f"{resource} and {reference} disagree — "
        f"only in the reference: {sorted(documented - deployed)}, "
        f"only in the template: {sorted(deployed - documented)}"
    )


def test_cfn_lint_is_clean():
    cfnlint_api = pytest.importorskip("cfnlint.api", reason="cfn-lint not installed")

    matches = cfnlint_api.lint(TEMPLATE.read_text())

    assert matches == [], "\n".join(str(match) for match in matches)
