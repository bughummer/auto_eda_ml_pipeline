"""
CanvasCostPoller

Runs every 5 minutes via EventBridge schedule.
- Reads active jobs from SSM
- Calculates accumulated cost per user
- Publishes custom CloudWatch metric: CanvasEstimatedCost per user
- Also checks Cost Explorer for daily actual spend (as backup)
- Triggers quarantine if either check exceeds budget
"""
import boto3
import json
from datetime import datetime, timezone, timedelta

ssm = boto3.client("ssm", region_name="us-east-1")
cloudwatch = boto3.client("cloudwatch", region_name="us-east-1")
sagemaker = boto3.client("sagemaker", region_name="us-east-1")
ce = boto3.client("ce", region_name="us-east-1")
iam = boto3.client("iam", region_name="us-east-1")

DOMAIN_ID = "d-yxvamwlr1iei"
GROUP_NAME = "Canvas_Users"
ROLE_NAME = "CanvasRestrictedExecutionRole"
NAMESPACE = "Canvas/CostTracking"

USER_BUDGET_LIMITS = {
    "Vasif_test_user": 100.0,
    "Aghamir_test_user": 100.0,
    "Murad_test_user": 100.0
}

BUDGET_USER_MAP = {
    "VasifBudget": "Vasif_test_user",
    "AghamirBudget": "Aghamir_test_user",
    "MuradBudget": "Murad_test_user"
}


# -------------------------
# SSM JOB TRACKING
# -------------------------
def get_active_jobs():
    """Get all active job tracking entries from SSM."""
    jobs = []
    try:
        paginator = ssm.get_paginator("get_parameters_by_path")
        for page in paginator.paginate(Path="/canvas/active-jobs/"):
            for param in page["Parameters"]:
                try:
                    jobs.append(json.loads(param["Value"]))
                except Exception:
                    pass
    except Exception as e:
        print(f"Failed to get active jobs from SSM: {e}")
    return jobs


def is_job_still_running(job):
    """Check if job is still in progress."""
    try:
        job_type = job["job_type"]
        job_name = job["job_name"]
        if job_type == "TrainingJob":
            resp = sagemaker.describe_training_job(TrainingJobName=job_name)
            return resp["TrainingJobStatus"] == "InProgress"
        elif job_type == "ProcessingJob":
            resp = sagemaker.describe_processing_job(ProcessingJobName=job_name)
            return resp["ProcessingJobStatus"] == "InProgress"
        elif job_type == "AutoMLJob":
            resp = sagemaker.describe_auto_ml_job(AutoMLJobName=job_name)
            return resp["AutoMLJobStatus"] == "InProgress"
        elif job_type == "TransformJob":
            resp = sagemaker.describe_transform_job(TransformJobName=job_name)
            return resp["TransformJobStatus"] == "InProgress"
    except Exception as e:
        print(f"Could not check job status for {job['job_name']}: {e}")
    return False


def remove_job_from_tracking(job_name):
    """Remove completed job from SSM."""
    try:
        ssm.delete_parameter(Name=f"/canvas/active-jobs/{job_name}")
        print(f"Removed completed job from tracking: {job_name}")
    except Exception as e:
        print(f"Failed to remove job {job_name} from SSM: {e}")


def update_job_cost(job, accumulated_cost):
    """Update accumulated cost in SSM."""
    try:
        job["accumulated_cost"] = accumulated_cost
        ssm.put_parameter(
            Name=f"/canvas/active-jobs/{job['job_name']}",
            Value=json.dumps(job),
            Type="String",
            Overwrite=True
        )
    except Exception as e:
        print(f"Failed to update job cost in SSM: {e}")


# -------------------------
# COST CALCULATION
# -------------------------
def calculate_job_cost(job):
    """Calculate current accumulated cost based on elapsed time."""
    try:
        start_time = datetime.fromisoformat(job["start_time"])
        now = datetime.now(timezone.utc)
        elapsed_hours = (now - start_time).total_seconds() / 3600
        cost = elapsed_hours * job["hourly_cost"]
        return round(cost, 4)
    except Exception as e:
        print(f"Error calculating cost for {job['job_name']}: {e}")
        return 0.0


# -------------------------
# CLOUDWATCH METRICS
# -------------------------
JOB_TYPE_LABEL = {
    "TrainingJob": "Training",
    "ProcessingJob": "Processing",
    "AutoMLJob": "Canvas",
    "TransformJob": "Transform",
}


def publish_user_cost_metric(user, estimated_cost, budget_limit, user_job_costs=None):
    """Publish estimated cost and budget % used to CloudWatch, with per job-type breakdown."""
    try:
        budget_pct = (estimated_cost / budget_limit * 100) if budget_limit > 0 else 0
        now = datetime.now(timezone.utc)

        metric_data = [
            {
                "MetricName": "EstimatedActiveCost",
                "Dimensions": [{"Name": "User", "Value": user}],
                "Value": estimated_cost,
                "Unit": "None",
                "Timestamp": now
            },
            {
                "MetricName": "BudgetUsedPercent",
                "Dimensions": [{"Name": "User", "Value": user}],
                "Value": budget_pct,
                "Unit": "Percent",
                "Timestamp": now
            }
        ]

        # Per job-type breakdown metrics
        if user_job_costs:
            for job_type, cost in user_job_costs.items():
                label = JOB_TYPE_LABEL.get(job_type, job_type)
                metric_data.append({
                    "MetricName": "EstimatedActiveCost",
                    "Dimensions": [
                        {"Name": "User", "Value": user},
                        {"Name": "JobType", "Value": label}
                    ],
                    "Value": cost,
                    "Unit": "None",
                    "Timestamp": now
                })

        cloudwatch.put_metric_data(Namespace=NAMESPACE, MetricData=metric_data)
        print(f"Published metric: {user} active cost = ${estimated_cost:.4f} ({budget_pct:.1f}% of budget)")
        if user_job_costs:
            for jt, c in user_job_costs.items():
                print(f"  {JOB_TYPE_LABEL.get(jt, jt)}: ${c:.4f}")
    except Exception as e:
        print(f"Failed to publish CloudWatch metric for {user}: {e}")


# -------------------------
# COST EXPLORER — ACTUAL SPEND BY SERVICE
# -------------------------
def get_ce_service_spend():
    """
    Get actual monthly SageMaker spend broken down by usage type.
    Returns dict: {service_label: amount}
    Covers Canvas sessions, Studio volume, training jobs etc.
    Note: 8-24h delay in Cost Explorer data.
    """
    today = datetime.now(timezone.utc).date()
    start_of_month = today.replace(day=1).isoformat()
    tomorrow = (today + timedelta(days=1)).isoformat()
    result = {}
    try:
        response = ce.get_cost_and_usage(
            TimePeriod={"Start": start_of_month, "End": tomorrow},
            Granularity="MONTHLY",
            Filter={"Dimensions": {"Key": "SERVICE", "Values": ["Amazon SageMaker"]}},
            Metrics=["UnblendedCost"],
            GroupBy=[{"Type": "DIMENSION", "Key": "USAGE_TYPE"}]
        )
        for group in response["ResultsByTime"][0]["Groups"]:
            usage_type = group["Keys"][0]
            amount = float(group["Metrics"]["UnblendedCost"]["Amount"])
            if amount > 0.0001:
                # Map usage types to readable labels
                if "Canvas" in usage_type:
                    label = "CanvasSession"
                elif "Studio" in usage_type:
                    label = "StudioVolume"
                elif "Train" in usage_type:
                    label = "Training"
                elif "Transform" in usage_type:
                    label = "Transform"
                elif "Processing" in usage_type:
                    label = "Processing"
                else:
                    label = usage_type.split(":")[-1] if ":" in usage_type else usage_type
                result[label] = result.get(label, 0.0) + amount
        print(f"CE actual spend by service: {result}")
    except Exception as e:
        print(f"Cost Explorer service spend failed: {e}")
    return result


def publish_actual_spend_metrics(service_costs, total_actual):
    """Publish actual CE spend to CloudWatch as account-level metrics."""
    try:
        now = datetime.now(timezone.utc)
        metric_data = [
            {
                "MetricName": "ActualMonthlySpend",
                "Dimensions": [{"Name": "Service", "Value": "Total"}],
                "Value": total_actual,
                "Unit": "None",
                "Timestamp": now
            }
        ]
        for service, amount in service_costs.items():
            metric_data.append({
                "MetricName": "ActualMonthlySpend",
                "Dimensions": [{"Name": "Service", "Value": service}],
                "Value": amount,
                "Unit": "None",
                "Timestamp": now
            })
        cloudwatch.put_metric_data(Namespace=NAMESPACE, MetricData=metric_data)
        print(f"Published actual spend metrics: Total=${total_actual:.4f}")
    except Exception as e:
        print(f"Failed to publish actual spend metrics: {e}")


# -------------------------
# COST EXPLORER BACKUP CHECK — PER USER
# -------------------------
def get_ce_actual_spend(username):
    """Get actual monthly spend from Cost Explorer (backup check, 8-24h delay)."""
    today = datetime.now(timezone.utc).date()
    start_of_month = today.replace(day=1).isoformat()
    tomorrow = (today + timedelta(days=1)).isoformat()
    try:
        response = ce.get_cost_and_usage(
            TimePeriod={"Start": start_of_month, "End": tomorrow},
            Granularity="MONTHLY",
            Filter={"Tags": {"Key": "Owner", "Values": [username]}},
            Metrics=["UnblendedCost"]
        )
        amount = float(response["ResultsByTime"][0]["Total"]["UnblendedCost"]["Amount"])
        print(f"Cost Explorer actual spend for {username}: ${amount:.4f}")
        return amount
    except Exception as e:
        print(f"Cost Explorer check failed for {username}: {e}")
        return None


# -------------------------
# QUARANTINE
# -------------------------
def is_already_quarantined(user):
    try:
        policies = iam.list_user_policies(UserName=user)["PolicyNames"]
        return "Quarantine" in policies
    except Exception:
        return False


def quarantine_user(user, reason):
    """Apply quarantine - stop jobs and block access."""
    print(f"QUARANTINING {user} — reason: {reason}")
    quarantine_policy = {
        "Version": "2012-10-17",
        "Statement": [{"Sid": "Quarantine", "Effect": "Deny", "Action": "*", "Resource": "*"}]
    }

    # Deny all on user
    try:
        iam.put_user_policy(
            UserName=user,
            PolicyName="Quarantine",
            PolicyDocument=json.dumps(quarantine_policy)
        )
        print(f"Quarantine attached to user: {user}")
    except Exception as e:
        print(f"Failed quarantine on user: {e}")

    # Deny all on role
    try:
        iam.put_role_policy(
            RoleName=ROLE_NAME,
            PolicyName="Quarantine",
            PolicyDocument=json.dumps(quarantine_policy)
        )
    except Exception as e:
        print(f"Failed quarantine on role: {e}")

    # Remove from group
    try:
        iam.remove_user_from_group(GroupName=GROUP_NAME, UserName=user)
    except Exception as e:
        print(f"Failed to remove from group: {e}")

    # Stop all running jobs
    stop_all_jobs()

    # Delete Canvas apps
    stop_canvas_apps()


def stop_all_jobs():
    for job_type, list_fn, stop_fn, name_key, status_key in [
        ("Training", sagemaker.list_training_jobs, sagemaker.stop_training_job,
         "TrainingJobName", "TrainingJobStatus"),
        ("Processing", sagemaker.list_processing_jobs, sagemaker.stop_processing_job,
         "ProcessingJobName", "ProcessingJobStatus"),
    ]:
        try:
            jobs = list_fn(StatusEquals="InProgress")[f"{job_type}JobSummaries"]
            for job in jobs:
                try:
                    stop_fn(**{name_key: job[name_key]})
                    print(f"Stopped {job_type} job: {job[name_key]}")
                except Exception as e:
                    print(f"Failed to stop {job[name_key]}: {e}")
        except Exception as e:
            print(f"Failed to list {job_type} jobs: {e}")

    # AutoML
    try:
        jobs = sagemaker.list_auto_ml_jobs(StatusEquals="InProgress")["AutoMLJobSummaries"]
        for job in jobs:
            try:
                sagemaker.stop_auto_ml_job(AutoMLJobName=job["AutoMLJobName"])
                print(f"Stopped AutoML job: {job['AutoMLJobName']}")
            except Exception as e:
                print(f"Failed to stop AutoML job: {e}")
    except Exception as e:
        print(f"Failed to list AutoML jobs: {e}")


def stop_canvas_apps():
    try:
        apps = sagemaker.list_apps(DomainIdEquals=DOMAIN_ID)["Apps"]
        for app in apps:
            if app["Status"] not in ["Deleted", "Deleting"]:
                try:
                    kwargs = {
                        "DomainId": DOMAIN_ID,
                        "AppType": app["AppType"],
                        "AppName": app["AppName"]
                    }
                    if "UserProfileName" in app:
                        kwargs["UserProfileName"] = app["UserProfileName"]
                    sagemaker.delete_app(**kwargs)
                    print(f"Deleted app: {app['AppName']}")
                except Exception as e:
                    print(f"Failed to delete app {app['AppName']}: {e}")
    except Exception as e:
        print(f"Failed to list apps: {e}")


# -------------------------
# MAIN HANDLER
# -------------------------
def scan_live_canvas_jobs():
    """
    Directly scan SageMaker API for active Canvas AutoML V2 jobs.
    This catches jobs that EventBridge may have missed (e.g. AutoML V2).
    Returns list of job dicts in the same format as SSM-tracked jobs.
    """
    live_jobs = []
    try:
        paginator = sagemaker.get_paginator("list_auto_ml_jobs")
        for page in paginator.paginate(StatusEquals="InProgress"):
            for job in page["AutoMLJobSummaries"]:
                job_name = job["AutoMLJobName"]
                try:
                    ssm.get_parameter(Name=f"/canvas/active-jobs/{job_name}")
                    continue
                except ssm.exceptions.ParameterNotFound:
                    pass
                user = guess_user_from_job_name(job_name)
                start_time = job.get("CreationTime", datetime.now(timezone.utc))
                if hasattr(start_time, 'isoformat'):
                    start_time_str = start_time.isoformat()
                else:
                    start_time_str = datetime.now(timezone.utc).isoformat()
                job_data = {
                    "job_name": job_name,
                    "job_type": "AutoMLJob",
                    "instance_type": "ml.m5.2xlarge",
                    "hourly_cost": INSTANCE_PRICES.get("ml.m5.2xlarge", 0.46),
                    "user": user,
                    "start_time": start_time_str,
                    "accumulated_cost": 0.0
                }
                try:
                    ssm.put_parameter(
                        Name=f"/canvas/active-jobs/{job_name}",
                        Value=json.dumps(job_data),
                        Type="String",
                        Overwrite=True
                    )
                    print(f"Late-registered Canvas job: {job_name} | user: {user}")
                    try:
                        sagemaker.add_tags(
                            ResourceArn=f"arn:aws:sagemaker:us-east-1:857753985214:automl-job/{job_name}",
                            Tags=[{"Key": "Owner", "Value": user}]
                        )
                    except Exception as e:
                        print(f"Failed to tag {job_name}: {e}")
                except Exception as e:
                    print(f"Failed to register {job_name} in SSM: {e}")
                live_jobs.append(job_data)
    except Exception as e:
        print(f"Failed to scan live Canvas jobs: {e}")
    return live_jobs


def scan_completed_canvas_jobs():
    """
    Scan for recently completed Canvas AutoML V2 jobs that EventBridge may have missed.
    Triggers CanvasAutoRetrain Lambda for any unprocessed completed jobs.
    """
    try:
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        paginator = sagemaker.get_paginator("list_auto_ml_jobs")
        for page in paginator.paginate(StatusEquals="Completed"):
            for job in page["AutoMLJobSummaries"]:
                job_name = job["AutoMLJobName"]
                end_time = job.get("EndTime")
                # Skip very old jobs (older than 30 days)
                if end_time and end_time < cutoff:
                    continue
                # Check if retrain already triggered (SSM marker)
                marker_key = f"/canvas/retrain-triggered/{job_name}"
                try:
                    ssm.get_parameter(Name=marker_key)
                    continue  # already triggered
                except ssm.exceptions.ParameterNotFound:
                    pass
                # Get user from tags
                try:
                    tags = sagemaker.list_tags(
                        ResourceArn=f"arn:aws:sagemaker:us-east-1:857753985214:automl-job/{job_name}"
                    ).get("Tags", [])
                    tag_dict = {t["Key"]: t["Value"] for t in tags}
                    profile_arn = tag_dict.get("sagemaker:user-profile-arn", "")
                    profile_name = profile_arn.split("/")[-1] if profile_arn else ""
                    profile_map = {
                        "Vasif-test-user": "Vasif_test_user",
                        "Aghamir-test-user": "Aghamir_test_user",
                        "Murad-test-user": "Murad_test_user",
                    }
                    user = profile_map.get(profile_name, "unknown")
                except Exception:
                    user = "unknown"

                print(f"Found unprocessed completed Canvas job: {job_name} | user: {user}")

                # Trigger CanvasAutoRetrain Lambda
                try:
                    import boto3 as _boto3
                    lam = _boto3.client("lambda", region_name="us-east-1")
                    import json as _json
                    lam.invoke(
                        FunctionName="CanvasAutoRetrain",
                        InvocationType="Event",
                        Payload=_json.dumps({"automl_job_name": job_name, "user": user}).encode()
                    )
                    print(f"Triggered CanvasAutoRetrain for {job_name}")
                except Exception as e:
                    print(f"Failed to trigger retrain for {job_name}: {e}")

                # Publish CompletedJobCost metric
                try:
                    end_time = job.get("EndTime")
                    create_time = job.get("CreationTime")
                    if end_time and create_time:
                        elapsed_hours = (end_time - create_time).total_seconds() / 3600
                        canvas_hourly = 0.46  # ml.m5.2xlarge estimate for Canvas AutoML
                        cost = round(elapsed_hours * canvas_hourly, 4)
                        cloudwatch.put_metric_data(
                            Namespace=NAMESPACE,
                            MetricData=[{
                                "MetricName": "CompletedJobCost",
                                "Dimensions": [
                                    {"Name": "User", "Value": user},
                                    {"Name": "JobType", "Value": "Canvas"}
                                ],
                                "Value": cost,
                                "Unit": "None",
                                "Timestamp": datetime.now(timezone.utc)
                            }]
                        )
                        print(f"Published CompletedJobCost: {user} | Canvas | {job_name} = ${cost:.4f}")
                except Exception as e:
                    print(f"Failed to publish CompletedJobCost for {job_name}: {e}")

                # Mark as triggered
                try:
                    ssm.put_parameter(
                        Name=marker_key,
                        Value=datetime.now(timezone.utc).isoformat(),
                        Type="String",
                        Overwrite=True
                    )
                except Exception:
                    pass
    except Exception as e:
        print(f"Failed to scan completed Canvas jobs: {e}")
    """
    Directly scan SageMaker API for active Canvas AutoML V2 jobs.
    This catches jobs that EventBridge may have missed (e.g. AutoML V2).
    Returns list of job dicts in the same format as SSM-tracked jobs.
    """
    live_jobs = []
    try:
        paginator = sagemaker.get_paginator("list_auto_ml_jobs")
        for page in paginator.paginate(StatusEquals="InProgress"):
            for job in page["AutoMLJobSummaries"]:
                job_name = job["AutoMLJobName"]
                # Check if already tracked in SSM
                try:
                    ssm.get_parameter(Name=f"/canvas/active-jobs/{job_name}")
                    continue  # already tracked
                except ssm.exceptions.ParameterNotFound:
                    pass

                # Try to get user from Canvas job tags first
                try:
                    tags = sagemaker.list_tags(
                        ResourceArn=f"arn:aws:sagemaker:us-east-1:857753985214:automl-job/{job_name}"
                    ).get("Tags", [])
                    tag_dict = {t["Key"]: t["Value"] for t in tags}
                    profile_arn = tag_dict.get("sagemaker:user-profile-arn", "")
                    if profile_arn:
                        profile_name = profile_arn.split("/")[-1]
                        profile_map = {
                            "Vasif-test-user": "Vasif_test_user",
                            "Vasif_test_user": "Vasif_test_user",
                            "Aghamir-test-user": "Aghamir_test_user",
                            "Aghamir_test_user": "Aghamir_test_user",
                            "Murad-test-user": "Murad_test_user",
                            "Murad_test_user": "Murad_test_user",
                        }
                        if profile_name in profile_map:
                            user = profile_map[profile_name]
                        else:
                            for u in USER_BUDGET_LIMITS.keys():
                                if u.lower().split("_")[0] in profile_name.lower():
                                    user = u
                                    break
                except Exception:
                    pass
                start_time = job.get("CreationTime", datetime.now(timezone.utc))
                if hasattr(start_time, 'isoformat'):
                    start_time_str = start_time.isoformat()
                else:
                    start_time_str = datetime.now(timezone.utc).isoformat()

                job_data = {
                    "job_name": job_name,
                    "job_type": "AutoMLJob",
                    "instance_type": "ml.m5.2xlarge",
                    "hourly_cost": INSTANCE_PRICES.get("ml.m5.2xlarge", 0.46),
                    "user": user,
                    "start_time": start_time_str,
                    "accumulated_cost": 0.0
                }
                # Register in SSM for future polls
                try:
                    ssm.put_parameter(
                        Name=f"/canvas/active-jobs/{job_name}",
                        Value=json.dumps(job_data),
                        Type="String",
                        Overwrite=True
                    )
                    print(f"Late-registered Canvas job: {job_name} | user: {user}")
                    # Tag with Owner for Cost Explorer tracking
                    try:
                        account = "857753985214"
                        region = "us-east-1"
                        sagemaker.add_tags(
                            ResourceArn=f"arn:aws:sagemaker:{region}:{account}:automl-job/{job_name}",
                            Tags=[{"Key": "Owner", "Value": user}]
                        )
                        print(f"Tagged Canvas job {job_name} with Owner={user}")
                    except Exception as e:
                        print(f"Failed to tag {job_name}: {e}")
                except Exception as e:
                    print(f"Failed to register {job_name} in SSM: {e}")
                live_jobs.append(job_data)
    except Exception as e:
        print(f"Failed to scan live Canvas jobs: {e}")
    return live_jobs


def guess_user_from_job_name(job_name):
    """Try to identify user from job name prefix."""
    for user in USER_BUDGET_LIMITS.keys():
        first_name = user.lower().split("_")[0]
        if first_name in job_name.lower():
            return user
    return "unknown"


def lambda_handler(event, context):
    print("Cost poller running...")

    # Scan for any live Canvas/AutoML jobs not yet in SSM
    scan_live_canvas_jobs()

    # Scan for recently completed Canvas jobs that EventBridge may have missed
    scan_completed_canvas_jobs()

    # Aggregate active job costs per user and per job type
    user_active_costs = {}
    user_job_type_costs = {}  # {user: {job_type: cost}}
    active_jobs = get_active_jobs()

    for job in active_jobs:
        user = job.get("user", "unknown")
        if user == "unknown":
            continue

        if not is_job_still_running(job):
            remove_job_from_tracking(job["job_name"])
            continue

        cost = calculate_job_cost(job)
        update_job_cost(job, cost)
        job_type = job.get("job_type", "Unknown")

        user_active_costs[user] = user_active_costs.get(user, 0.0) + cost

        if user not in user_job_type_costs:
            user_job_type_costs[user] = {}
        user_job_type_costs[user][job_type] = user_job_type_costs[user].get(job_type, 0.0) + cost

    # For each tracked user, publish metrics and check limits
    for user, budget_limit in USER_BUDGET_LIMITS.items():
        active_cost = user_active_costs.get(user, 0.0)
        job_costs = user_job_type_costs.get(user, {})

        print(f"{user}: active job cost=${active_cost:.4f} / limit=${budget_limit}")

        # Publish to CloudWatch with per job-type breakdown
        publish_user_cost_metric(user, active_cost, budget_limit, job_costs)

        # Check if over budget
        if active_cost >= budget_limit:
            if not is_already_quarantined(user):
                quarantine_user(
                    user,
                    f"Estimated total ${active_cost:.2f} >= limit ${budget_limit:.2f}"
                )
            else:
                print(f"{user} already quarantined")

    # Publish actual CE spend by service (Canvas sessions, Studio, Training etc.)
    service_costs = get_ce_service_spend()
    total_actual = sum(service_costs.values())
    if service_costs:
        publish_actual_spend_metrics(service_costs, total_actual)

    return {"status": "DONE", "users_checked": list(USER_BUDGET_LIMITS.keys())}
