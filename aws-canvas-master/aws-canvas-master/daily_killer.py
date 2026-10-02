import boto3
import time

sagemaker = boto3.client("sagemaker", region_name="us-east-1")

DOMAIN_ID = "d-yxvamwlr1iei"


def stop_all_running_jobs():
    # Training jobs
    try:
        paginator = sagemaker.get_paginator("list_training_jobs")
        for page in paginator.paginate(StatusEquals="InProgress"):
            for job in page["TrainingJobSummaries"]:
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
        for page in paginator.paginate(StatusEquals="InProgress"):
            for job in page["ProcessingJobSummaries"]:
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
        for page in paginator.paginate(StatusEquals="InProgress"):
            for job in page["AutoMLJobSummaries"]:
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
        for page in paginator.paginate(StatusEquals="InProgress"):
            for job in page["TransformJobSummaries"]:
                try:
                    sagemaker.stop_transform_job(TransformJobName=job["TransformJobName"])
                    print(f"Stopped transform job: {job['TransformJobName']}")
                except Exception as e:
                    print(f"Failed to stop transform job {job['TransformJobName']}: {e}")
    except Exception as e:
        print(f"Failed to list transform jobs: {e}")


def stop_all_canvas_apps():
    try:
        paginator = sagemaker.get_paginator("list_apps")
        for page in paginator.paginate(DomainIdEquals=DOMAIN_ID):
            for app in page["Apps"]:
                if app["Status"] not in ["Deleted", "Deleting"]:
                    try:
                        kwargs = {
                            "DomainId": DOMAIN_ID,
                            "AppType": app["AppType"],
                            "AppName": app["AppName"]
                        }
                        if "UserProfileName" in app:
                            kwargs["UserProfileName"] = app["UserProfileName"]
                        if "SpaceName" in app:
                            kwargs["SpaceName"] = app["SpaceName"]
                        sagemaker.delete_app(**kwargs)
                        print(f"Deleted app: {app['AppName']} (type: {app['AppType']})")
                    except Exception as e:
                        print(f"Failed to delete app {app['AppName']}: {e}")
    except Exception as e:
        print(f"Failed to list apps: {e}")

    print("Waiting for apps to shut down...")
    time.sleep(30)


def lambda_handler(event, context):
    print("Daily 7 PM kill switch triggered (Azerbaijan time / 15:00 UTC)")

    stop_all_running_jobs()
    stop_all_canvas_apps()

    print("All running processes stopped.")
    return {"status": "DONE"}
