"""
CanvasAutoRetrain Lambda

Triggered by CanvasRealtimeCostTracker when a Canvas AutoML job completes.

Pipeline:
  1. Read Canvas model name from job tags → derive project bucket name
  2. Copy Canvas preprocessed data → bucket/datasets/train_sample/
  3. Start Processing Job → bucket/datasets/train_full/ + bucket/datasets/validation/
  4. Wait for Processing Job to complete
  5. Start XGBoost Training Job → bucket/model/candidate/
  6. Write canvas feasibility metrics → bucket/metrics_artifacts/canvas_feasibility_metrics.json

Naming convention:
  Canvas model name : cvm-churn-prepaid-30d
  Project bucket    : azc-ml-prj-cvm-churn-prepaid-30d

Input event:
  {
    "automl_job_name": "Canvas1784208029563",
    "user": "Vasif_test_user",
    "canvas_data_path": "s3://.../.../preprocessed-data/tuning_data/train/"
  }
"""
import boto3
import json
import time
from datetime import datetime, timezone

sagemaker = boto3.client("sagemaker", region_name="us-east-1")
s3 = boto3.client("s3", region_name="us-east-1")

ACCOUNT = "857753985214"
REGION = "us-east-1"
EXECUTION_ROLE = f"arn:aws:iam::{ACCOUNT}:role/service-role/AmazonSageMaker-ExecutionRole-20260513T164883"
FALLBACK_BUCKET = "canvas-datasets-vorujzada"

# SageMaker images (us-east-1)
XGBOOST_IMAGE = f"683313688378.dkr.ecr.{REGION}.amazonaws.com/sagemaker-xgboost:1.3-1"
SKLEARN_IMAGE = f"683313688378.dkr.ecr.{REGION}.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3"
AUTOGLUON_TRAINING_IMAGE = f"763104351884.dkr.ecr.{REGION}.amazonaws.com/autogluon-training:1.1.0-cpu-py310-ubuntu20.04"
PREPROCESS_SCRIPT_S3 = f"s3://{FALLBACK_BUCKET}/scripts/preprocess.py"


def uid():
    return datetime.now(timezone.utc).strftime("%m%d%H%M%S")


# -------------------------
# AUTOML JOB METADATA
# -------------------------
def describe_automl_job(automl_job_name):
    """Describe AutoML job — tries V2 first (Canvas uses V2)."""
    try:
        return sagemaker.describe_auto_ml_job_v2(AutoMLJobName=automl_job_name)
    except Exception:
        return sagemaker.describe_auto_ml_job(AutoMLJobName=automl_job_name)


def get_job_tags(automl_job_name):
    """Get tags dict from AutoML job."""
    try:
        arn = f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:automl-job/{automl_job_name}"
        tags = sagemaker.list_tags(ResourceArn=arn).get("Tags", [])
        return {t["Key"]: t["Value"] for t in tags}
    except Exception as e:
        print(f"Could not get tags for {automl_job_name}: {e}")
        return {}


def get_project_bucket(automl_job_name):
    """
    Derive project bucket from Canvas model name.
    Canvas model name 'cvm-churn-prepaid-30d' → bucket 'azc-ml-prj-cvm-churn-prepaid-30d'
    If Canvas model name starts with 'Model_' (auto-generated), strip it.
    If no usable model name found, fall back to FALLBACK_BUCKET.
    """
    tags = get_job_tags(automl_job_name)
    canvas_model_name = tags.get("CanvasModelName", "")

    if canvas_model_name:
        # Strip auto-generated prefix if present
        if canvas_model_name.startswith("Model_"):
            canvas_model_name = canvas_model_name[6:]
        # Clean for S3 bucket name: lowercase, replace underscores/spaces with hyphens
        clean = canvas_model_name.lower().replace("_", "-").replace(" ", "-")
        # Remove any non-alphanumeric-hyphen chars
        clean = "".join(c for c in clean if c.isalnum() or c == "-")
        if clean:
            bucket = f"azc-ml-prj-{clean}"
            # Verify bucket exists
            try:
                s3.head_bucket(Bucket=bucket)
                print(f"Project bucket found: {bucket}")
                return bucket
            except Exception:
                print(f"Bucket {bucket} does not exist — falling back to {FALLBACK_BUCKET}")

    print(f"No project bucket found, using fallback: {FALLBACK_BUCKET}")
    return FALLBACK_BUCKET


def get_canvas_data_path(automl_job_name):
    """Find the preprocessed train data path from the Canvas AutoML job."""
    try:
        resp = describe_automl_job(automl_job_name)
        output_path = resp.get("OutputDataConfig", {}).get("S3OutputPath", "").rstrip("/")
        if output_path:
            return f"{output_path}/{automl_job_name}/preprocessed-data/tuning_data/train/"
    except Exception as e:
        print(f"Could not get Canvas data path: {e}")
    return None


def get_num_classes(automl_job_name):
    """Detect number of classes from Canvas AutoML job."""
    try:
        resp = describe_automl_job(automl_job_name)
        problem_type = (
            resp.get("ProblemType", "") or
            resp.get("AutoMLProblemTypeConfig", {}).get("TabularJobConfig", {}).get("ProblemType", "")
        )
        if problem_type == "BinaryClassification":
            return 2
        elif problem_type == "MulticlassClassification":
            return 4
        elif problem_type == "Regression":
            return 1
    except Exception as e:
        print(f"Could not get problem type: {e}")
    return 2


# -------------------------
# S3 STRUCTURE
# -------------------------
def ensure_bucket_structure(bucket):
    """Create placeholder objects to establish the project folder structure."""
    folders = [
        "canvas/README",
        "config/README",
        "datasets/train_sample/README",
        "datasets/train_full/README",
        "datasets/validation/README",
        "datasets/test/README",
        "datasets/scoring/README",
        "model/candidate/README",
        "model/approved/README",
        "model/archived/README",
        "metrics_artifacts/README",
        "batch_outputs/predictions/README",
        "batch_outputs/rejected_records/README",
    ]
    for key in folders:
        try:
            s3.put_object(Bucket=bucket, Key=key, Body=b"")
        except Exception as e:
            print(f"Could not create {key} in {bucket}: {e}")
    print(f"Folder structure ensured in s3://{bucket}/")


def copy_canvas_sample_data(canvas_data_path, project_bucket, automl_job_name):
    """Copy Canvas preprocessed data to bucket/datasets/train_sample/."""
    try:
        src_bucket = canvas_data_path.split("/")[2]
        src_prefix = "/".join(canvas_data_path.split("/")[3:])
        paginator = s3.get_paginator("list_objects_v2")
        copied = 0
        for page in paginator.paginate(Bucket=src_bucket, Prefix=src_prefix):
            for obj in page.get("Contents", []):
                src_key = obj["Key"]
                filename = src_key.split("/")[-1]
                dst_key = f"datasets/train_sample/{filename}"
                s3.copy_object(
                    CopySource={"Bucket": src_bucket, "Key": src_key},
                    Bucket=project_bucket,
                    Key=dst_key
                )
                copied += 1
        print(f"Copied {copied} files to s3://{project_bucket}/datasets/train_sample/")
    except Exception as e:
        print(f"Could not copy Canvas sample data: {e}")


def write_canvas_metrics(project_bucket, automl_job_name):
    """Write Canvas feasibility metrics to bucket/metrics_artifacts/."""
    try:
        resp = describe_automl_job(automl_job_name)
        best_candidate = resp.get("BestCandidate", {})
        metrics = {
            "automl_job_name": automl_job_name,
            "status": resp.get("AutoMLJobStatus", ""),
            "problem_type": (
                resp.get("ProblemType", "") or
                resp.get("AutoMLProblemTypeConfig", {}).get("TabularJobConfig", {}).get("ProblemType", "")
            ),
            "best_candidate": best_candidate.get("CandidateName", ""),
            "final_metrics": best_candidate.get("FinalAutoMLJobObjectiveMetric", {}),
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        s3.put_object(
            Bucket=project_bucket,
            Key="metrics_artifacts/canvas_feasibility_metrics.json",
            Body=json.dumps(metrics, indent=2, default=str).encode()
        )
        print(f"Canvas metrics written to s3://{project_bucket}/metrics_artifacts/canvas_feasibility_metrics.json")
    except Exception as e:
        print(f"Could not write Canvas metrics: {e}")


# -------------------------
# PROCESSING JOB
# -------------------------
def start_processing_job(automl_job_name, user, train_data_path, project_bucket):
    """Start Processing Job — output goes to project bucket structure."""
    job_name = f"retrain-preprocess-{uid()}"
    train_out = f"s3://{project_bucket}/datasets/train_full/data/"
    val_out = f"s3://{project_bucket}/datasets/validation/data/"

    print(f"Starting Processing Job: {job_name}")
    print(f"  Input:  {train_data_path}")
    print(f"  Train:  {train_out}")
    print(f"  Val:    {val_out}")

    sagemaker.create_processing_job(
        ProcessingJobName=job_name,
        ProcessingResources={
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": "ml.m5.xlarge",
                "VolumeSizeInGB": 20
            }
        },
        AppSpecification={
            "ImageUri": SKLEARN_IMAGE,
            "ContainerEntrypoint": ["python3", "/opt/ml/processing/code/preprocess.py"]
        },
        ProcessingInputs=[
            {
                "InputName": "train-data",
                "S3Input": {
                    "S3Uri": train_data_path,
                    "LocalPath": "/opt/ml/processing/input",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File"
                }
            },
            {
                "InputName": "code",
                "S3Input": {
                    "S3Uri": PREPROCESS_SCRIPT_S3,
                    "LocalPath": "/opt/ml/processing/code",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File"
                }
            }
        ],
        ProcessingOutputConfig={
            "Outputs": [
                {
                    "OutputName": "train",
                    "S3Output": {
                        "S3Uri": train_out,
                        "LocalPath": "/opt/ml/processing/output/train",
                        "S3UploadMode": "EndOfJob"
                    }
                },
                {
                    "OutputName": "validation",
                    "S3Output": {
                        "S3Uri": val_out,
                        "LocalPath": "/opt/ml/processing/output/validation",
                        "S3UploadMode": "EndOfJob"
                    }
                }
            ]
        },
        RoleArn=EXECUTION_ROLE,
        Tags=[
            {"Key": "Owner", "Value": user},
            {"Key": "SourceCanvas", "Value": automl_job_name}
        ]
    )

    return job_name, train_out, val_out


# -------------------------
# TRAINING JOB
# -------------------------
def get_canvas_best_hyperparameters(automl_job_name):
    """
    Get hyperparameters from Canvas BestCandidate training job.
    Returns dict of hyperparameters to pass to the full training job.
    """
    try:
        resp = describe_automl_job(automl_job_name)
        best = resp.get("BestCandidate", {})
        steps = best.get("CandidateSteps", [])
        for step in steps:
            if step.get("CandidateStepType") == "AWS::SageMaker::TrainingJob":
                train_job_name = step["CandidateStepName"]
                train_resp = sagemaker.describe_training_job(TrainingJobName=train_job_name)
                hps = train_resp.get("HyperParameters", {})
                print(f"Canvas best candidate HPs: {list(hps.keys())[:10]}")
                return hps, train_resp.get("AlgorithmSpecification", {}).get("TrainingImage", "")
    except Exception as e:
        print(f"Could not get Canvas best HPs: {e}")
    return {}, ""


def start_training_job(automl_job_name, user, train_out, val_out, project_bucket, num_classes):
    """
    Start Training Job using the SAME algorithm and hyperparameters as Canvas BestCandidate.
    If Canvas used AutoGluon, uses AutoGluon training image with same HPs.
    Falls back to XGBoost if AutoGluon image not available.
    """
    job_name = f"retrain-{uid()}"
    model_output = f"s3://{project_bucket}/model/candidate/"

    print(f"Starting Training Job: {job_name}")

    # Get Canvas best candidate's algorithm + hyperparameters
    best_hps, canvas_image = get_canvas_best_hyperparameters(automl_job_name)

    # Determine training image
    if "autogluon" in canvas_image.lower():
        training_image = AUTOGLUON_TRAINING_IMAGE
        print(f"Using AutoGluon image (same as Canvas)")
        # AutoGluon hyperparameters — use Canvas best candidate's HPs
        # Key AutoGluon HPs Canvas sets
        hyperparameters = {
            k: v for k, v in best_hps.items()
            if k in ["eval_metric", "problem_type", "presets", "time_limit",
                     "excluded_model_types", "auto_stack", "num_bag_folds",
                     "num_bag_sets", "num_stack_levels"]
        }
        # Increase time_limit for full dataset training
        if "time_limit" in hyperparameters:
            try:
                orig = int(hyperparameters["time_limit"])
                hyperparameters["time_limit"] = str(min(orig * 3, 3600))  # 3x but max 1hr
                print(f"Increased time_limit: {orig}s -> {hyperparameters['time_limit']}s")
            except Exception:
                hyperparameters["time_limit"] = "1800"
        else:
            hyperparameters["time_limit"] = "1800"
    else:
        # Fallback: XGBoost
        training_image = XGBOOST_IMAGE
        print(f"Using XGBoost image (Canvas image not recognized)")
        if num_classes == 1:
            hyperparameters = {"num_round": "100", "objective": "reg:squarederror", "eval_metric": "rmse", "max_depth": "6", "eta": "0.3", "subsample": "0.8", "colsample_bytree": "0.8"}
        elif num_classes == 2:
            hyperparameters = {"num_round": "100", "objective": "binary:logistic", "eval_metric": "logloss", "max_depth": "6", "eta": "0.3", "subsample": "0.8", "colsample_bytree": "0.8"}
        else:
            hyperparameters = {"num_round": "100", "objective": "multi:softprob", "num_class": str(num_classes), "eval_metric": "mlogloss", "max_depth": "6", "eta": "0.3", "subsample": "0.8", "colsample_bytree": "0.8"}

    print(f"Training image: ...{training_image[-50:]}")
    print(f"Hyperparameters: {hyperparameters}")

    sagemaker.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            "TrainingImage": training_image,
            "TrainingInputMode": "File"
        },
        RoleArn=EXECUTION_ROLE,
        InputDataConfig=[
            {
                "ChannelName": "train",
                "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": train_out, "S3DataDistributionType": "FullyReplicated"}},
                "ContentType": "text/csv",
                "InputMode": "File"
            },
            {
                "ChannelName": "validation",
                "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": val_out, "S3DataDistributionType": "FullyReplicated"}},
                "ContentType": "text/csv",
                "InputMode": "File"
            }
        ],
        OutputDataConfig={"S3OutputPath": model_output},
        ResourceConfig={"InstanceType": "ml.m5.xlarge", "InstanceCount": 1, "VolumeSizeInGB": 20},
        StoppingCondition={"MaxRuntimeInSeconds": 3600},
        HyperParameters=hyperparameters,
        Tags=[
            {"Key": "Owner", "Value": user},
            {"Key": "SourceCanvas", "Value": automl_job_name}
        ]
    )

    return job_name, model_output


def publish_job_cost(user, job_name, job_type_label, cost_usd):
    """Publish CompletedJobCost metric to CloudWatch."""
    try:
        from datetime import datetime, timezone
        cloudwatch.put_metric_data(
            Namespace="Canvas/CostTracking",
            MetricData=[{
                "MetricName": "CompletedJobCost",
                "Dimensions": [
                    {"Name": "User", "Value": user},
                    {"Name": "JobType", "Value": job_type_label}
                ],
                "Value": cost_usd,
                "Unit": "None",
                "Timestamp": datetime.now(timezone.utc)
            }]
        )
        print(f"Published CompletedJobCost: {user} | {job_type_label} = ${cost_usd:.4f}")
    except Exception as e:
        print(f"Failed to publish CompletedJobCost for {job_name}: {e}")


def write_training_metrics(project_bucket, training_job_name):
    """Poll training job and write final metrics to bucket/metrics_artifacts/training_metrics.json."""
    try:
        resp = sagemaker.describe_training_job(TrainingJobName=training_job_name)
        metrics = {
            "training_job_name": training_job_name,
            "status": resp.get("TrainingJobStatus", ""),
            "instance_type": resp.get("ResourceConfig", {}).get("InstanceType", ""),
            "final_metrics": [
                {"name": m["MetricName"], "value": m["Value"]}
                for m in resp.get("FinalMetricDataList", [])
            ],
            "model_artifacts": resp.get("ModelArtifacts", {}).get("S3ModelArtifacts", ""),
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        s3.put_object(
            Bucket=project_bucket,
            Key="metrics_artifacts/training_metrics.json",
            Body=json.dumps(metrics, indent=2, default=str).encode()
        )
        print(f"Training metrics written to s3://{project_bucket}/metrics_artifacts/training_metrics.json")
    except Exception as e:
        print(f"Could not write training metrics: {e}")


# -------------------------
# BATCH TRANSFORM
# -------------------------
def prepare_scoring_data_from_test(project_bucket):
    """
    Copy datasets/validation/ data to datasets/scoring/ with target column removed.
    Validation data has target in col 0 — strip it for scoring.
    """
    src_prefix = "datasets/validation/"
    dst_prefix = "datasets/scoring/"

    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=project_bucket, Prefix=src_prefix):
            for obj in page.get("Contents", []):
                src_key = obj["Key"]
                if src_key.endswith(".keep") or src_key == src_prefix:
                    continue
                filename = src_key.split("/")[-1]
                if not filename:
                    continue

                # Download, strip target column (col 0), re-upload
                response = s3.get_object(Bucket=project_bucket, Key=src_key)
                content = response["Body"].read().decode("utf-8")
                lines = content.strip().split("\n")
                stripped = []
                for line in lines:
                    if not line.strip():
                        continue
                    cols = line.split(",")
                    # Remove first column (target)
                    stripped.append(",".join(cols[1:]))

                dst_key = f"{dst_prefix}{filename}"
                s3.put_object(
                    Bucket=project_bucket,
                    Key=dst_key,
                    Body="\n".join(stripped).encode("utf-8")
                )
                print(f"Scoring data prepared: s3://{project_bucket}/{dst_key} ({len(stripped)} rows)")
        return f"s3://{project_bucket}/{dst_prefix}"
    except Exception as e:
        print(f"Could not prepare scoring data: {e}")
        return None


def prepare_scoring_from_validation(project_bucket):
    """
    If no real scoring data exists, use validation data with target column (col 0) removed.
    Writes to datasets/scoring/scoring_from_validation.csv
    Returns the S3 URI to use for Batch Transform input.
    """
    try:
        # Find validation data
        val_prefix = "datasets/validation/data/"
        resp = s3.list_objects_v2(Bucket=project_bucket, Prefix=val_prefix, MaxKeys=5)
        val_files = [o for o in resp.get("Contents", []) if o["Key"].endswith(".csv") and o["Size"] > 0]

        if not val_files:
            # Try old path
            val_prefix = "datasets/validation/"
            resp = s3.list_objects_v2(Bucket=project_bucket, Prefix=val_prefix, MaxKeys=5)
            val_files = [o for o in resp.get("Contents", []) if o["Key"].endswith(".csv") and o["Size"] > 0]

        if not val_files:
            print("No validation data found to prepare scoring file")
            return None

        src_key = val_files[0]["Key"]
        print(f"Preparing scoring data from {src_key}")

        # Download, strip target column (col 0), re-upload
        response = s3.get_object(Bucket=project_bucket, Key=src_key)
        content = response["Body"].read().decode("utf-8")
        lines = [l for l in content.strip().split("\n") if l.strip()]
        stripped = []
        for line in lines:
            cols = line.split(",")
            stripped.append(",".join(cols[1:]))  # remove target column

        dst_key = "datasets/scoring/scoring_from_validation.csv"
        s3.put_object(
            Bucket=project_bucket,
            Key=dst_key,
            Body="\n".join(stripped).encode("utf-8")
        )
        scoring_uri = f"s3://{project_bucket}/datasets/scoring/"
        print(f"Scoring file prepared: s3://{project_bucket}/{dst_key} ({len(stripped)} rows)")
        return scoring_uri
    except Exception as e:
        print(f"Failed to prepare scoring from validation: {e}")
        return None


def create_sagemaker_model(training_job_name, user, automl_job_name):
    """Create a SageMaker Model object from Training Job artifacts."""
    model_name = f"model-{training_job_name}"
    try:
        resp = sagemaker.describe_training_job(TrainingJobName=training_job_name)
        model_artifact = resp["ModelArtifacts"]["S3ModelArtifacts"]
        training_image = resp["AlgorithmSpecification"]["TrainingImage"]

        sagemaker.create_model(
            ModelName=model_name,
            ExecutionRoleArn=EXECUTION_ROLE,
            PrimaryContainer={
                "Image": training_image,
                "ModelDataUrl": model_artifact
            },
            Tags=[
                {"Key": "Owner", "Value": user},
                {"Key": "SourceCanvas", "Value": automl_job_name}
            ]
        )
        print(f"SageMaker Model created: {model_name}")
        return model_name
    except Exception as e:
        print(f"Failed to create SageMaker Model: {e}")
        return None


def start_batch_transform(model_name, scoring_input, project_bucket, user, automl_job_name):
    """Start Batch Transform job — predictions go to batch_outputs/predictions/."""
    job_name = f"batch-{uid()}"
    output_path = f"s3://{project_bucket}/batch_outputs/predictions/"

    print(f"Starting Batch Transform: {job_name}")
    print(f"  Input:  {scoring_input}")
    print(f"  Output: {output_path}")

    sagemaker.create_transform_job(
        TransformJobName=job_name,
        ModelName=model_name,
        TransformInput={
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": scoring_input
                }
            },
            "ContentType": "text/csv",
            "SplitType": "Line"
        },
        TransformOutput={
            "S3OutputPath": output_path,
            "Accept": "text/csv",
            "AssembleWith": "Line"
        },
        TransformResources={
            "InstanceType": "ml.m5.xlarge",
            "InstanceCount": 1
        },
        MaxConcurrentTransforms=1,
        MaxPayloadInMB=6,
        Tags=[
            {"Key": "Owner", "Value": user},
            {"Key": "SourceCanvas", "Value": automl_job_name}
        ]
    )
    return job_name, output_path


# -------------------------
# MAIN HANDLER
# -------------------------
def lambda_handler(event, context):
    print("Auto-retrain triggered:", json.dumps(event))

    automl_job_name = event.get("automl_job_name")
    user = event.get("user", "unknown")
    canvas_data_path = event.get("canvas_data_path")

    if not automl_job_name:
        print("No automl_job_name in event")
        return {"status": "ERROR", "reason": "missing automl_job_name"}

    # Step 1: Find project bucket from Canvas model name
    project_bucket = get_project_bucket(automl_job_name)
    print(f"Project bucket: s3://{project_bucket}/")

    # Step 2: Ensure folder structure exists
    ensure_bucket_structure(project_bucket)

    # Step 3: Get Canvas data path
    if not canvas_data_path:
        canvas_data_path = get_canvas_data_path(automl_job_name)
    if not canvas_data_path:
        print("Could not determine Canvas data path")
        return {"status": "ERROR", "reason": "cannot find canvas data path"}

    # Step 4: Copy Canvas sample data to datasets/train_sample/
    copy_canvas_sample_data(canvas_data_path, project_bucket, automl_job_name)

    # Step 5: Write Canvas feasibility metrics
    write_canvas_metrics(project_bucket, automl_job_name)

    num_classes = get_num_classes(automl_job_name)
    print(f"Problem: {num_classes} classes | User: {user}")

    # Step 6: Start Processing Job
    try:
        proc_job_name, train_out, val_out = start_processing_job(
            automl_job_name, user, canvas_data_path, project_bucket
        )
        print(f"Processing Job started: {proc_job_name}")
    except Exception as e:
        print(f"Failed to start Processing Job: {e}")
        return {"status": "ERROR", "reason": str(e)}

    # Step 7: Wait for Processing Job
    print(f"Waiting for Processing Job {proc_job_name}...")
    for _ in range(90):
        resp = sagemaker.describe_processing_job(ProcessingJobName=proc_job_name)
        proc_status = resp["ProcessingJobStatus"]
        print(f"  Processing: {proc_status}")
        if proc_status == "Completed":
            break
        elif proc_status in ("Failed", "Stopped"):
            reason = resp.get("FailureReason", "unknown")
            print(f"Processing failed: {reason}")
            return {"status": "PROCESSING_FAILED", "reason": reason}
        time.sleep(10)
    else:
        return {"status": "PROCESSING_TIMEOUT"}

    # Publish Processing cost
    try:
        proc_resp = sagemaker.describe_processing_job(ProcessingJobName=proc_job_name)
        proc_start = proc_resp.get("ProcessingStartTime")
        proc_end = proc_resp.get("ProcessingEndTime")
        if proc_start and proc_end:
            elapsed = (proc_end - proc_start).total_seconds() / 3600
            proc_cost = round(elapsed * 0.23, 4)  # ml.m5.xlarge
            publish_job_cost(user, proc_job_name, "Processing", proc_cost)
    except Exception as e:
        print(f"Could not publish Processing cost: {e}")

    # Step 8: Start Training Job
    try:
        train_job_name, model_output = start_training_job(
            automl_job_name, user, train_out, val_out, project_bucket, num_classes
        )
        print(f"Training Job started: {train_job_name}")
    except Exception as e:
        print(f"Failed to start Training Job: {e}")
        return {"status": "TRAINING_FAILED", "error": str(e)}

    # Step 9: Wait for Training Job and write metrics
    print(f"Waiting for Training Job {train_job_name}...")
    for _ in range(180):
        resp = sagemaker.describe_training_job(TrainingJobName=train_job_name)
        train_status = resp["TrainingJobStatus"]
        print(f"  Training: {train_status}")
        if train_status == "Completed":
            write_training_metrics(project_bucket, train_job_name)
            # Publish Training cost
            try:
                tr = sagemaker.describe_training_job(TrainingJobName=train_job_name)
                t_start = tr.get("TrainingStartTime")
                t_end = tr.get("TrainingEndTime")
                if t_start and t_end:
                    elapsed = (t_end - t_start).total_seconds() / 3600
                    train_cost = round(elapsed * 0.23, 4)
                    publish_job_cost(user, train_job_name, "Training", train_cost)
            except Exception as e:
                print(f"Could not publish Training cost: {e}")
            break
        elif train_status in ("Failed", "Stopped"):
            reason = resp.get("FailureReason", "unknown")
            print(f"Training failed: {reason}")
            return {"status": "TRAINING_FAILED", "reason": reason}
        time.sleep(10)
    else:
        return {"status": "TRAINING_TIMEOUT"}

    # Step 10: Check if scoring data exists → start Batch Transform
    scoring_input = f"s3://{project_bucket}/datasets/scoring/"
    batch_job_name = None
    try:
        resp = s3.list_objects_v2(Bucket=project_bucket, Prefix="datasets/scoring/", MaxKeys=10)
        # Filter out placeholder files (README, .keep, empty)
        scoring_files = [
            o for o in resp.get("Contents", [])
            if not o["Key"].endswith("README")
            and not o["Key"].endswith(".keep")
            and o["Size"] > 0
        ]

        if scoring_files:
            print(f"Found {len(scoring_files)} real scoring file(s) — starting Batch Transform")
        else:
            # No real scoring data — use validation data with target column removed
            print("No scoring data found — preparing scoring file from validation data")
            scoring_input = prepare_scoring_from_validation(project_bucket)

        if scoring_input:
            model_name = create_sagemaker_model(train_job_name, user, automl_job_name)
            if model_name:
                batch_job_name, batch_output = start_batch_transform(
                    model_name, scoring_input, project_bucket, user, automl_job_name
                )
                print(f"Batch Transform started: {batch_job_name}")
                # Wait for batch and publish cost
                for _ in range(60):
                    b_resp = sagemaker.describe_transform_job(TransformJobName=batch_job_name)
                    b_status = b_resp["TransformJobStatus"]
                    if b_status == "Completed":
                        b_start = b_resp.get("TransformStartTime")
                        b_end = b_resp.get("TransformEndTime")
                        if b_start and b_end:
                            elapsed = (b_end - b_start).total_seconds() / 3600
                            batch_cost = round(elapsed * 0.23, 4)
                            publish_job_cost(user, batch_job_name, "Transform", batch_cost)
                        break
                    elif b_status in ("Failed", "Stopped"):
                        print(f"Batch Transform failed: {b_resp.get('FailureReason','')}")
                        break
                    time.sleep(10)
    except Exception as e:
        print(f"Batch Transform step failed: {e}")

    return {
        "status": "PIPELINE_COMPLETED",
        "project_bucket": f"s3://{project_bucket}/",
        "automl_job": automl_job_name,
        "processing_job": proc_job_name,
        "training_job": train_job_name,
        "model_output": model_output,
        "batch_transform_job": batch_job_name,
        "user": user
    }
