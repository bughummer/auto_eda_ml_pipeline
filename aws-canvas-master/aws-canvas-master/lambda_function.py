import boto3
import json
import time
from datetime import datetime, timezone, timedelta

iam = boto3.client("iam")
sagemaker = boto3.client("sagemaker")
ce = boto3.client("ce")

GROUP_NAME = "Canvas_Users"
DOMAIN_ID = "d-yxvamwlr1iei"
ROLE_NAME = "CanvasRestrictedExecutionRole"

BUDGET_USER_MAP = {
    "VasifBudget": "Vasif_test_user",
    "AghamirBudget": "Aghamir_test_user",
    "MuradBudget": "Murad_test_user"
}

# Budget limits per user in USD
USER_BUDGET_LIMITS = {
    "Vasif_test_user": 100.0,
    "Aghamir_test_user": 100.0,
    "Murad_test_user": 100.0
}


# -------------------------
# GET ACTUAL SPEND PER USER
# Uses CreatedBy tag on resources (set by SageMaker automatically)
# -------------------------
def get_user_actual_spend(username):
    """
    Query Cost Explorer for spend tagged with Owner=username.
    Falls back to returning None if tags aren't active yet.
    """
    today = datetime.now(timezone.utc).date()
    start_of_month = today.replace(day=1).isoformat()
    tomorrow = (today + timedelta(days=1)).isoformat()

    try:
        response = ce.get_cost_and_usage(
            TimePeriod={"Start": start_of_month, "End": tomorrow},
            Granularity="MONTHLY",
            Filter={
                "Tags": {
                    "Key": "Owner",
                    "Values": [username]
                }
            },
            Metrics=["UnblendedCost"]
        )
        amount = float(response["ResultsByTime"][0]["Total"]["UnblendedCost"]["Amount"])
        print(f"Actual spend for {username} (by Owner tag): ${amount:.4f}")
        return amount
    except Exception as e:
        print(f"Could not get tagged spend for {username}: {e}")
        return None


def is_over_budget(username):
    """Check if user is over their individual budget limit."""
    limit = USER_BUDGET_LIMITS.get(username)
    if not limit:
        print(f"No budget limit defined for {username}, defaulting to enforce")
        return True

    spend = get_user_actual_spend(username)
    if spend is None:
        # Can't verify per-user spend — trust the SNS budget alert
        print(f"Cannot verify per-user spend for {username}, trusting SNS alert")
        return True

    over = spend >= limit
    print(f"{username} spend: ${spend:.2f} / limit: ${limit:.2f} — {'OVER' if over else 'OK'}")
    return over


# -------------------------
# PARSE SNS MESSAGE
# -------------------------
def parse_budget_name(message):
    try:
        msg = json.loads(message)
        return msg.get("budgetName") or msg.get("Budget Name")
    except json.JSONDecodeError:
        for line in message.split("\n"):
            if "Budget Name:" in line:
                return line.split("Budget Name:")[1].strip()
    return None


def is_breach(message):
    lower = message.lower()
    return "exceeded" in lower or "greater than" in lower


def is_already_quarantined(user):
    try:
        policies = iam.list_user_policies(UserName=user)["PolicyNames"]
        return "Quarantine" in policies
    except Exception as e:
        print(f"Could not check quarantine status for {user}: {e}")
        return False


# -------------------------
# IAM QUARANTINE
# -------------------------
def attach_quarantine(user):
    quarantine_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "Quarantine",
                "Effect": "Deny",
                "Action": "*",
                "Resource": "*"
            }
        ]
    }

    try:
        iam.put_user_policy(
            UserName=user,
            PolicyName="Quarantine",
            PolicyDocument=json.dumps(quarantine_policy)
        )
        print(f"Quarantine attached to user: {user}")
    except Exception as e:
        print(f"Failed to attach quarantine to user {user}: {e}")

    try:
        iam.put_role_policy(
            RoleName=ROLE_NAME,
            PolicyName="Quarantine",
            PolicyDocument=json.dumps(quarantine_policy)
        )
        print(f"Quarantine attached to role: {ROLE_NAME}")
    except Exception as e:
        print(f"Failed to attach quarantine to role {ROLE_NAME}: {e}")


# -------------------------
# REVOKE ACTIVE SESSIONS
# -------------------------
def revoke_active_sessions():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    revoke_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "RevokeOldSessions",
                "Effect": "Deny",
                "Action": "*",
                "Resource": "*",
                "Condition": {
                    "DateLessThan": {
                        "aws:TokenIssueTime": now
                    }
                }
            }
        ]
    }
    try:
        iam.put_role_policy(
            RoleName=ROLE_NAME,
            PolicyName="RevokeActiveSessions",
            PolicyDocument=json.dumps(revoke_policy)
        )
        print(f"Active sessions revoked for role: {ROLE_NAME}")
    except Exception as e:
        print(f"Failed to revoke sessions for role {ROLE_NAME}: {e}")


# -------------------------
# STOP SAGEMAKER JOBS
# -------------------------
def stop_all_jobs(user):
    # Training jobs
    try:
        paginator = sagemaker.get_paginator("list_training_jobs")
        for page in paginator.paginate(NameContains=user):
            for job in page["TrainingJobSummaries"]:
                if job["TrainingJobStatus"] == "InProgress":
                    try:
                        sagemaker.stop_training_job(TrainingJobName=job["TrainingJobName"])
                        print(f"Stopped training job: {job['TrainingJobName']}")
                    except Exception as e:
                        print(f"Failed to stop training job {job['TrainingJobName']}: {e}")
    except Exception as e:
        print(f"Failed to list training jobs: {e}")

    # Processing jobs
    try:
        paginator = sagemaker.get_paginator("list_processing_jobs")
        for page in paginator.paginate(NameContains=user):
            for job in page["ProcessingJobSummaries"]:
                if job["ProcessingJobStatus"] == "InProgress":
                    try:
                        sagemaker.stop_processing_job(ProcessingJobName=job["ProcessingJobName"])
                        print(f"Stopped processing job: {job['ProcessingJobName']}")
                    except Exception as e:
                        print(f"Failed to stop processing job {job['ProcessingJobName']}: {e}")
    except Exception as e:
        print(f"Failed to list processing jobs: {e}")

    # AutoML jobs
    try:
        paginator = sagemaker.get_paginator("list_auto_ml_jobs")
        for page in paginator.paginate(NameContains=user):
            for job in page["AutoMLJobSummaries"]:
                if job["AutoMLJobStatus"] == "InProgress":
                    try:
                        sagemaker.stop_auto_ml_job(AutoMLJobName=job["AutoMLJobName"])
                        print(f"Stopped AutoML job: {job['AutoMLJobName']}")
                    except Exception as e:
                        print(f"Failed to stop AutoML job {job['AutoMLJobName']}: {e}")
    except Exception as e:
        print(f"Failed to list AutoML jobs: {e}")

    # Transform jobs
    try:
        paginator = sagemaker.get_paginator("list_transform_jobs")
        for page in paginator.paginate(NameContains=user):
            for job in page["TransformJobSummaries"]:
                if job["TransformJobStatus"] == "InProgress":
                    try:
                        sagemaker.stop_transform_job(TransformJobName=job["TransformJobName"])
                        print(f"Stopped transform job: {job['TransformJobName']}")
                    except Exception as e:
                        print(f"Failed to stop transform job {job['TransformJobName']}: {e}")
    except Exception as e:
        print(f"Failed to list transform jobs: {e}")

    # Also stop any unscoped running jobs (account-wide safety net)
    try:
        jobs = sagemaker.list_processing_jobs(StatusEquals="InProgress")["ProcessingJobSummaries"]
        for job in jobs:
            try:
                sagemaker.stop_processing_job(ProcessingJobName=job["ProcessingJobName"])
                print(f"Stopped processing job (global): {job['ProcessingJobName']}")
            except Exception as e:
                print(f"Failed stopping processing job {job['ProcessingJobName']}: {e}")
    except Exception as e:
        print(f"Failed to list global processing jobs: {e}")

    try:
        jobs = sagemaker.list_training_jobs(StatusEquals="InProgress")["TrainingJobSummaries"]
        for job in jobs:
            try:
                sagemaker.stop_training_job(TrainingJobName=job["TrainingJobName"])
                print(f"Stopped training job (global): {job['TrainingJobName']}")
            except Exception as e:
                print(f"Failed stopping training job {job['TrainingJobName']}: {e}")
    except Exception as e:
        print(f"Failed to list global training jobs: {e}")


# -------------------------
# STOP CANVAS APPS AND SPACES
# -------------------------
def stop_canvas_and_spaces():
    try:
        apps = sagemaker.list_apps(DomainIdEquals=DOMAIN_ID)["Apps"]
        for app in apps:
            if app["Status"] not in ["Deleted", "Deleting"]:
                try:
                    sagemaker.delete_app(
                        DomainId=DOMAIN_ID,
                        UserProfileName=app["UserProfileName"],
                        AppType=app["AppType"],
                        AppName=app["AppName"]
                    )
                    print(f"Deleted app {app['AppName']} for {app['UserProfileName']}")
                except Exception as e:
                    print(f"Failed to delete app {app['AppName']}: {e}")
    except Exception as e:
        print(f"Failed to list apps: {e}")

    print("Waiting for apps to delete...")
    time.sleep(30)

    try:
        spaces = sagemaker.list_spaces(DomainIdEquals=DOMAIN_ID)["Spaces"]
        for space in spaces:
            if space["Status"] not in ["Deleted", "Deleting"]:
                try:
                    sagemaker.delete_space(
                        DomainId=DOMAIN_ID,
                        SpaceName=space["SpaceName"]
                    )
                    print(f"Deleted space: {space['SpaceName']}")
                except Exception as e:
                    print(f"Failed to delete space {space['SpaceName']}: {e}")
    except Exception as e:
        print(f"Failed to list spaces: {e}")

    print("Waiting for spaces to delete...")
    time.sleep(15)


# -------------------------
# DELETE USER PROFILE
# -------------------------
def delete_user_profile(user):
    try:
        sagemaker.delete_user_profile(
            DomainId=DOMAIN_ID,
            UserProfileName=user
        )
        print(f"Deleted user profile: {user}")
    except Exception as e:
        print(f"Failed to delete user profile {user}: {e}")


# -------------------------
# MAIN HANDLER
# -------------------------
def lambda_handler(event, context):
    message = event["Records"][0]["Sns"]["Message"]
    print("RAW SNS:", message)

    budget_name = parse_budget_name(message)
    print(f"Budget name: {budget_name}")

    user = BUDGET_USER_MAP.get(budget_name)
    if not user:
        print(f"No mapped user for budget: {budget_name}")
        return {"status": "NO_USER"}

    if not is_breach(message):
        print("Not a breach alert, no action taken")
        return {"status": "NO_ACTION"}

    # Double-check actual per-user spend before enforcing
    if not is_over_budget(user):
        print(f"{user} is not actually over budget — skipping enforcement")
        return {"status": "NO_ACTION", "reason": "per_user_spend_ok"}

    if is_already_quarantined(user):
        print(f"{user} is already quarantined")
        return {"status": "ALREADY_QUARANTINED", "user": user}

    try:
        print(f"Enforcing budget for {user}...")

        # 1. Remove from group
        try:
            iam.remove_user_from_group(GroupName=GROUP_NAME, UserName=user)
            print(f"Removed {user} from group {GROUP_NAME}")
        except Exception as e:
            print(f"Could not remove from group: {e}")

        # 2. Attach quarantine deny-all to user and role
        attach_quarantine(user)

        # 3. Revoke all active sessions
        revoke_active_sessions()

        # 4. Stop all running SageMaker jobs
        stop_all_jobs(user)

        # 5. Delete Canvas apps and spaces
        stop_canvas_and_spaces()

        # 6. Delete user profile
        delete_user_profile(user)

        return {
            "status": "QUARANTINED",
            "user": user,
            "budget": budget_name
        }

    except Exception as e:
        print(f"ERROR: {str(e)}")
        return {
            "status": "ERROR",
            "error": str(e)
        }
