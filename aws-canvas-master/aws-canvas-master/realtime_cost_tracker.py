"""
CanvasRealtimeCostTracker

Triggered by EventBridge when a SageMaker job starts OR completes.
- On start: stores job metadata in SSM for the poller to pick up
- On complete: calculates final cost and publishes CompletedJobCost metric to CloudWatch

Tracks: Canvas (AutoML), Training, Processing, Transform jobs — per user, per job type.
"""
import boto3
import json
from datetime import datetime, timezone

ssm = boto3.client("ssm", region_name="us-east-1")
sagemaker = boto3.client("sagemaker", region_name="us-east-1")
cloudwatch = boto3.client("cloudwatch", region_name="us-east-1")
lambda_client = boto3.client("lambda", region_name="us-east-1")

AUTO_RETRAIN_FUNCTION = "CanvasAutoRetrain"

NAMESPACE = "Canvas/CostTracking"

# Known hourly prices for allowed instance types (us-east-1)
INSTANCE_PRICES = {
    "ml.t3.micro":   0.0280,
    "ml.t3.small":   0.0460,
    "ml.t3.medium":  0.0550,
    "ml.t3.large":   0.1020,
    "ml.t3.xlarge":  0.2040,
    "ml.t3.2xlarge": 0.4080,
    "ml.m5.xlarge":  0.2300,
    "ml.m5.2xlarge": 0.4600,
    "ml.m5.4xlarge": 0.9200,
    "ml.c5.xlarge":  0.2040,
    "ml.c5.2xlarge": 0.4080,
}

# Maps IAM usernames to SageMaker user profile names
USER_PROFILE_MAP = {
    "Vasif_test_user": "Vasif_test_user",
    "Aghamir_test_user": "Aghamir_test_user",
    "Murad_test_user": "Murad_test_user"
}

USER_BUDGET_LIMITS = {
    "Vasif_test_user": 100.0,
    "Aghamir_test_user": 100.0,
    "Murad_test_user": 100.0
}

# Terminal statuses — job is done
TERMINAL_STATUSES = {"Completed", "Failed", "Stopped", "Cancelled"}

# Active statuses — job is running
ACTIVE_STATUSES = {"InProgress", "Starting"}


def get_job_instance_type(job_type, job_name):
    """Get instance type from job details."""
    try:
        if job_type == "TrainingJob":
            resp = sagemaker.describe_training_job(TrainingJobName=job_name)
            return resp.get("ResourceConfig", {}).get("InstanceType", "ml.m5.xlarge")
        elif job_type == "ProcessingJob":
            resp = sagemaker.describe_processing_job(ProcessingJobName=job_name)
            return resp.get("ProcessingResources", {}).get("ClusterConfig", {}).get("InstanceType", "ml.m5.xlarge")
        elif job_type == "AutoMLJob":
            # Canvas AutoML uses managed instances — estimate with m5.2xlarge
            return "ml.m5.2xlarge"
        elif job_type == "TransformJob":
            resp = sagemaker.describe_transform_job(TransformJobName=job_name)
            return resp.get("TransformResources", {}).get("InstanceType", "ml.m5.xlarge")
    except Exception as e:
        print(f"Could not get instance type for {job_name}: {e}")
    return "ml.m5.xlarge"


def get_instance_hourly_cost(instance_type):
    """Look up hourly price for instance type."""
    return INSTANCE_PRICES.get(instance_type, 0.50)


def extract_job_info(event):
    """Extract job name, type, status, and user from EventBridge event."""
    detail = event.get("detail", {})
    detail_type = event.get("detail-type", "")
    job_type = None
    job_name = None
    status = None

    if "TrainingJobName" in detail:
        job_type = "TrainingJob"
        job_name = detail["TrainingJobName"]
        status = detail.get("TrainingJobStatus", "")
    elif "ProcessingJobName" in detail:
        job_type = "ProcessingJob"
        job_name = detail["ProcessingJobName"]
        status = detail.get("ProcessingJobStatus", "")
    elif "AutoMLJobName" in detail:
        job_type = "AutoMLJob"
        job_name = detail["AutoMLJobName"]
        # V1 and V2 both use AutoMLJobStatus
        status = detail.get("AutoMLJobStatus", "")
    elif "TransformJobName" in detail:
        job_type = "TransformJob"
        job_name = detail["TransformJobName"]
        status = detail.get("TransformJobStatus", "")

    return job_type, job_name, status


def guess_user_from_job_name(job_name):
    """Try to identify user from job name prefix."""
    for user in USER_PROFILE_MAP.keys():
        # Match first name part (e.g. "vasif" in "Vasif_test_user")
        first_name = user.lower().split("_")[0]
        if first_name in job_name.lower():
            return user
    return "unknown"


def get_user_from_job_tags(job_type, job_name):
    """
    Get the IAM username from SageMaker job tags.
    Canvas sets 'sagemaker:user-profile-arn' tag on every job it creates.
    Falls back to job name guessing if tags not available.
    """
    PROFILE_TO_USER = {
        "Vasif-test-user": "Vasif_test_user",
        "Vasif_test_user": "Vasif_test_user",
        "Aghamir-test-user": "Aghamir_test_user",
        "Aghamir_test_user": "Aghamir_test_user",
        "Murad-test-user": "Murad_test_user",
        "Murad_test_user": "Murad_test_user",
    }

    try:
        region = "us-east-1"
        account = "857753985214"
        arn_map = {
            "TrainingJob":   f"arn:aws:sagemaker:{region}:{account}:training-job/{job_name}",
            "ProcessingJob": f"arn:aws:sagemaker:{region}:{account}:processing-job/{job_name}",
            "AutoMLJob":     f"arn:aws:sagemaker:{region}:{account}:automl-job/{job_name}",
            "TransformJob":  f"arn:aws:sagemaker:{region}:{account}:transform-job/{job_name}",
        }
        resource_arn = arn_map.get(job_type)
        if resource_arn:
            tags = sagemaker.list_tags(ResourceArn=resource_arn).get("Tags", [])
            tag_dict = {t["Key"]: t["Value"] for t in tags}

            # Canvas sets sagemaker:user-profile-arn tag
            profile_arn = tag_dict.get("sagemaker:user-profile-arn", "")
            if profile_arn:
                profile_name = profile_arn.split("/")[-1]
                if profile_name in PROFILE_TO_USER:
                    return PROFILE_TO_USER[profile_name]
                # Try first name match in profile name
                for user in USER_PROFILE_MAP.keys():
                    first_name = user.lower().split("_")[0]
                    if first_name in profile_name.lower():
                        return user

            # Check Owner tag (manually set jobs)
            owner = tag_dict.get("Owner", "")
            if owner in USER_PROFILE_MAP:
                return owner
    except Exception as e:
        print(f"Could not read tags for {job_name}: {e}")

    # Fallback: job name
    return guess_user_from_job_name(job_name)


def job_type_label(job_type):
    """Return a clean label for CloudWatch dimension."""
    return {
        "TrainingJob": "Training",
        "ProcessingJob": "Processing",
        "AutoMLJob": "Canvas",
        "TransformJob": "Transform",
    }.get(job_type, job_type)


def publish_completed_job_cost(user, job_name, job_type, cost_usd):
    """Publish final cost of a completed job to CloudWatch."""
    try:
        cloudwatch.put_metric_data(
            Namespace=NAMESPACE,
            MetricData=[
                {
                    "MetricName": "CompletedJobCost",
                    "Dimensions": [
                        {"Name": "User", "Value": user},
                        {"Name": "JobType", "Value": job_type_label(job_type)},
                    ],
                    "Value": cost_usd,
                    "Unit": "None",
                    "Timestamp": datetime.now(timezone.utc)
                }
            ]
        )
        print(f"Published CompletedJobCost: {user} | {job_type_label(job_type)} | {job_name} = ${cost_usd:.4f}")
    except Exception as e:
        print(f"Failed to publish CompletedJobCost metric: {e}")


def tag_job_with_owner(job_type, job_name, user):
    """Add Owner tag to a SageMaker job so Cost Explorer can track per-user costs."""
    if user == "unknown":
        return
    try:
        region = "us-east-1"
        account = "857753985214"
        arn_map = {
            "TrainingJob":  f"arn:aws:sagemaker:{region}:{account}:training-job/{job_name}",
            "ProcessingJob": f"arn:aws:sagemaker:{region}:{account}:processing-job/{job_name}",
            "AutoMLJob":    f"arn:aws:sagemaker:{region}:{account}:automl-job/{job_name}",
            "TransformJob": f"arn:aws:sagemaker:{region}:{account}:transform-job/{job_name}",
        }
        resource_arn = arn_map.get(job_type)
        if not resource_arn:
            print(f"Unknown job type for tagging: {job_type}")
            return
        sagemaker.add_tags(
            ResourceArn=resource_arn,
            Tags=[{"Key": "Owner", "Value": user}]
        )
        print(f"Tagged {job_name} with Owner={user}")
    except Exception as e:
        print(f"Failed to tag {job_name}: {e}")


def handle_job_start(job_type, job_name, user):
    """Store job tracking info in SSM and tag job with Owner when job starts."""
    # Tag the job with Owner so Cost Explorer can track per-user costs
    tag_job_with_owner(job_type, job_name, user)

    instance_type = get_job_instance_type(job_type, job_name)
    hourly_cost = get_instance_hourly_cost(instance_type)
    start_time = datetime.now(timezone.utc).isoformat()

    param_name = f"/canvas/active-jobs/{job_name}"
    job_data = {
        "job_name": job_name,
        "job_type": job_type,
        "instance_type": instance_type,
        "hourly_cost": hourly_cost,
        "user": user,
        "start_time": start_time,
        "accumulated_cost": 0.0
    }

    try:
        ssm.put_parameter(
            Name=param_name,
            Value=json.dumps(job_data),
            Type="String",
            Overwrite=True
        )
        print(f"Tracking started: {job_name} | {job_type_label(job_type)} | {instance_type} | ${hourly_cost}/hr | user: {user}")
    except Exception as e:
        print(f"Failed to store job tracking in SSM: {e}")


def trigger_auto_retrain(job_name, user):
    """Invoke CanvasAutoRetrain Lambda when a Canvas AutoML job completes."""
    try:
        # Get Canvas data path from the completed job
        resp = sagemaker.describe_auto_ml_job(AutoMLJobName=job_name)
        output_path = resp.get("OutputDataConfig", {}).get("S3OutputPath", "").rstrip("/")
        canvas_data_path = f"{output_path}/{job_name}/preprocessed-data/tuning_data/train/"

        payload = {
            "automl_job_name": job_name,
            "user": user,
            "canvas_data_path": canvas_data_path
        }
        print(f"Triggering auto-retrain for {job_name} | user: {user}")
        lambda_client.invoke(
            FunctionName=AUTO_RETRAIN_FUNCTION,
            InvocationType="Event",  # async — fire and forget
            Payload=json.dumps(payload).encode()
        )
        print(f"Auto-retrain triggered successfully")
    except Exception as e:
        print(f"Failed to trigger auto-retrain for {job_name}: {e}")


def handle_job_complete(job_type, job_name, user):
    """Calculate final cost and publish metric when job completes."""
    param_name = f"/canvas/active-jobs/{job_name}"
    final_cost = 0.0

    try:
        param = ssm.get_parameter(Name=param_name)
        job_data = json.loads(param["Parameter"]["Value"])
        start_time = datetime.fromisoformat(job_data["start_time"])
        now = datetime.now(timezone.utc)
        elapsed_hours = (now - start_time).total_seconds() / 3600
        final_cost = round(elapsed_hours * job_data["hourly_cost"], 4)
        print(f"Job completed: {job_name} | elapsed: {elapsed_hours:.2f}h | cost: ${final_cost:.4f}")
    except ssm.exceptions.ParameterNotFound:
        # Job wasn't tracked from start — estimate from SageMaker describe
        print(f"No SSM record for {job_name}, estimating cost from SageMaker")
        instance_type = get_job_instance_type(job_type, job_name)
        hourly_cost = get_instance_hourly_cost(instance_type)
        try:
            if job_type == "TrainingJob":
                resp = sagemaker.describe_training_job(TrainingJobName=job_name)
                start = resp.get("TrainingStartTime")
                end = resp.get("TrainingEndTime")
            elif job_type == "ProcessingJob":
                resp = sagemaker.describe_processing_job(ProcessingJobName=job_name)
                start = resp.get("ProcessingStartTime")
                end = resp.get("ProcessingEndTime")
            elif job_type == "TransformJob":
                resp = sagemaker.describe_transform_job(TransformJobName=job_name)
                start = resp.get("TransformStartTime")
                end = resp.get("TransformEndTime")
            else:
                start = end = None

            if start and end:
                elapsed_hours = (end - start).total_seconds() / 3600
                final_cost = round(elapsed_hours * hourly_cost, 4)
        except Exception as e:
            print(f"Could not estimate cost from SageMaker for {job_name}: {e}")
    except Exception as e:
        print(f"Failed to read SSM for {job_name}: {e}")

    # Publish final cost metric
    publish_completed_job_cost(user, job_name, job_type, final_cost)

    # Clean up SSM
    try:
        ssm.delete_parameter(Name=param_name)
    except Exception:
        pass

    # Trigger auto-retrain pipeline when Canvas AutoML job completes
    if job_type == "AutoMLJob" and user != "unknown":
        trigger_auto_retrain(job_name, user)


def lambda_handler(event, context):
    print("Job event received:", json.dumps(event))

    job_type, job_name, status = extract_job_info(event)
    if not job_type or not job_name:
        print("Could not extract job info from event")
        return {"status": "NO_JOB"}

    user = get_user_from_job_tags(job_type, job_name)
    print(f"Job: {job_name} | Type: {job_type_label(job_type)} | Status: {status} | User: {user}")

    if status in ACTIVE_STATUSES:
        handle_job_start(job_type, job_name, user)
        return {"status": "TRACKING_STARTED", "job": job_name, "user": user}

    elif status in TERMINAL_STATUSES:
        handle_job_complete(job_type, job_name, user)
        return {"status": "TRACKING_COMPLETED", "job": job_name, "user": user}

    else:
        print(f"Unhandled status: {status} for {job_name}")
        return {"status": "SKIPPED", "job": job_name, "status_received": status}
