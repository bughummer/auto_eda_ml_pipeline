# Infrastructure

Everything AWS-side. No always-on ML compute: the only permanently running components are the
FastAPI control plane and the React app, both on the corporate server.

```
stepfunctions/   the two workflow definitions (ASL)
iam/             the three role boundaries, as standalone policy documents
cloudformation/  one stack: artifact bucket, KMS key, roles, workflows
docker/          the single job image used by every SageMaker job
```

## The two workflows

`eda_state_machine.asl.json` — profiling only. It writes `EDA_RUNNING` to the experiment's
`state/workflow.json` with the `s3:putObject` SDK integration, runs the profiling Processing
job and hands control back for feature review. The execution ends there
rather than waiting: feature review is human time, and an execution left open for days is
fragile for no benefit.

`training_state_machine.asl.json` — preparation, a `Map` over the configured models (one
SageMaker Training job each), then evaluation. The Map iterates the model list from the
execution input, so adding a model plugin never requires touching this file. Each branch has
its own `Catch`, so one model failing leaves the rest of the experiment intact; the evaluation
job reads each model's outcome from its artifacts and records the terminal experiment state.

Both definitions use CloudFormation `DefinitionSubstitutions` for the image URI, the job role,
instance sizing and the KMS key, so nothing environment-specific is committed.

## IAM boundaries

| Role | Trusted by | May do | May not do |
|---|---|---|---|
| `MlFactoryBackendRole` | the corporate server | start and describe executions, read/write the artifact prefix (records included), read approved datasets, invoke the allow-listed Bedrock model | create SageMaker jobs, touch any other bucket |
| `MlFactoryWorkflowRole` | `states.amazonaws.com` | create/describe/stop `mlf-*` jobs, pass the job role to SageMaker, write `…/experiments/*/state/*` | read datasets or artifacts directly |
| `MlFactoryJobRole` | `sagemaker.amazonaws.com` | read the approved dataset prefix, read/write one experiment artifact prefix, record the terminal state, write logs | start further jobs, reach other buckets |

The policy documents in `iam/` are the reference version with `${PLACEHOLDER}` variables; the
CloudFormation template contains the deployed equivalents with real ARNs.

## Deploying

`make deploy-aws` runs all of this for you (`scripts/deploy_aws.sh`); the steps are spelled out
here for reference.

```bash
# 1. Build and push the job image
docker build -f infrastructure/docker/Dockerfile -t ml-factory-jobs:1.0.0 .
docker tag ml-factory-jobs:1.0.0 "$ECR_REGISTRY/ml-factory-jobs:1.0.0"
docker push "$ECR_REGISTRY/ml-factory-jobs:1.0.0"

# 2. Upload the workflow definitions
aws s3 cp infrastructure/stepfunctions/ "s3://$DEFINITIONS_BUCKET/ml-factory/stepfunctions/" \
  --recursive --exclude "*" --include "*.asl.json"

# 3. Create or update the stack
aws cloudformation deploy \
  --template-file infrastructure/cloudformation/ml-factory.yaml \
  --stack-name ml-factory \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
      ArtifactBucketName="$ARTIFACT_BUCKET" \
      ApprovedDataBucketName="$APPROVED_DATA_BUCKET" \
      ApprovedDataPrefix=curated \
      JobImageUri="$ECR_REGISTRY/ml-factory-jobs:1.0.0" \
      DefinitionsBucket="$DEFINITIONS_BUCKET"

# 4. Feed the stack outputs into the control plane's environment (.env)
aws cloudformation describe-stacks --stack-name ml-factory \
  --query 'Stacks[0].Outputs' --output table
```

## Corporate proxy

The control plane builds every boto3 client through `backend/aws/clients.py`, which reads
`HTTPS_PROXY`/`HTTP_PROXY`/`NO_PROXY` and applies them to the botocore config explicitly rather
than relying on ambient behaviour. Set them in the service unit that runs the API. The job
containers need outbound access to S3 and CloudWatch only; if the training subnets
sit behind the proxy, pass the same variables through the job `Environment` block.

## Cost shape

Nothing runs between experiments. An experiment costs four short jobs plus one training job
per model, all on CPU instances, plus S3 storage for artifacts and records. There is no
database and nothing to pay for between experiments. Instance types and the parallelism cap
are stack parameters, not code.
