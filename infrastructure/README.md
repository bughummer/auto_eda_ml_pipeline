# Infrastructure

Everything AWS-side. There is no always-on AWS compute of any kind — not for ML, not for the
control plane. The only permanently running component is the `docker compose` container (FastAPI
control plane + built React app), and it runs on the corporate server. AWS is reached from there
over HTTPS; AWS never reaches back.

```
stepfunctions/   the two workflow definitions (ASL)
iam/             the three runtime role boundaries, plus the deployer's own permissions
cloudformation/  one stack: artifact bucket, KMS key, roles, the two workflows
docker/          the single job image used by every SageMaker job
```

Every SageMaker job here is **ephemeral**: Step Functions calls `CreateProcessingJob` or
`CreateTrainingJob`, AWS provisions an instance, pulls the job image, runs the entrypoint, writes
to S3 and destroys the instance. Nothing is left running to shut down, and nothing is billed
between experiments.

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

Two different kinds of "permissions" are easy to conflate. **Runtime roles** are what the
platform itself assumes while it runs — you never use these credentials yourself. The
**deployer's permissions** are what your own AWS identity (an IAM user, an SSO role, a CI role)
needs to stand the platform up in the first place. They're deliberately separate: the deployer
can create and configure the stack without being able to act as the running platform, and vice
versa.

### Runtime roles (the platform's own credentials)

| Role | Trusted by | May do | May not do |
|---|---|---|---|
| `MlFactoryBackendRole` | `sts:AssumeRole` from this account | start and describe executions, read/write the artifact prefix (records included), read approved datasets, invoke the allow-listed Bedrock model | create SageMaker jobs, touch any other bucket |
| `MlFactoryWorkflowRole` | `states.amazonaws.com` | create/describe/stop `mlf-*` jobs, pass the job role to SageMaker, write `…/experiments/*/state/*` | read datasets or artifacts directly |
| `MlFactoryJobRole` | `sagemaker.amazonaws.com` | read the approved dataset prefix, read/write one experiment artifact prefix, record the terminal state, write logs | start further jobs, reach other buckets |

The policy documents in `iam/backend_role_policy.json`, `workflow_role_policy.json` and
`job_role_policy.json` are the reference version with `${PLACEHOLDER}` variables; the
CloudFormation template contains the deployed equivalents with real ARNs. `MlFactoryBackendRole`
is what the control plane on the corporate server acts as — the IAM user whose keys go into
`config/secrets/config.py` assumes it — and the other two are assumed by AWS services, never by
you.

### Deployer permissions (your own AWS identity)

`iam/deployer_policy.json` is what to attach to whoever — or whatever CI role — runs
`make deploy-aws`. It covers exactly what that requires: create/update the CloudFormation stack;
create the three runtime roles (`CAPABILITY_NAMED_IAM`) and pass them to the services that use
them; create the artifact bucket and its KMS key; push to ECR; upload the workflow definitions;
and create the two state machines. It provisions no compute, because there is none to provision
— the control plane's host is yours. It is not `AdministratorAccess`, and it is not one of the
three runtime roles above: attaching it to an operator does not let that operator act as the
running platform, only stand it up.

If your organization requires a first deploy under a broader policy (e.g. while validating the
resource list), `deployer_policy.json` is still the target to narrow down to afterward — every
action in it traces to a specific step below.

### Supplying a role instead of letting the stack create it

Some organizations require every IAM role to be created and reviewed outside of CloudFormation.
Each of the three runtime roles can be supplied instead of created: pass its ARN via
`ExistingBackendRoleArn`, `ExistingWorkflowRoleArn`, or `ExistingJobRoleArn` (`make deploy-aws`
via `EXISTING_BACKEND_ROLE_ARN` etc., `make deploy-aws-nocli` the same, or
`--parameter-overrides` directly) and the stack references that role instead of creating
`MlFactoryBackendRole` / `MlFactoryWorkflowRole` / `MlFactoryJobRole`. Leave a parameter empty
(the default) to have the stack create that one role as usual — mixing is fine, each of the
three is independent.

The stack never attaches a policy to a role you supply, so it must already carry the matching
reference policy — `backend_role_policy.json`, `workflow_role_policy.json`, or
`job_role_policy.json` — with the placeholders filled in for this deployment, trusted the way
that policy's `Comment` describes (backend: `sts:AssumeRole` from the account; workflow: from
`states.amazonaws.com`; job: from `sagemaker.amazonaws.com`). Create and attach it before
deploying — the stack will reference the ARN, not validate what it can do.

## Deploying

`make deploy-aws` runs all of this for you (`scripts/deploy_aws.sh`, which shells out to the
`aws` CLI); the steps are spelled out here for reference.

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

### Without the `aws` CLI

`make deploy-aws-nocli` (`scripts/deploy_aws.py`) does the same four steps over boto3 instead —
no CLI binary, just Python and the `boto3` package (already installed with `make install`).
Docker is still required: building and pushing the job image has no SDK equivalent.

The same environment variables apply. Credentials, in order of preference:

1. **`DEPLOYER_ROLE_ARN`** — whatever base identity boto3 finds (env vars, a profile, an
   attached role) calls `sts:AssumeRole` on it, and every AWS call in the script runs as that
   role. The base identity only needs `sts:AssumeRole` on this one ARN; the role itself needs
   `deployer_policy.json`. Use this when you were handed a role ARN to deploy as, rather than
   your own long-lived credentials.
2. **`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`** (plus `AWS_SESSION_TOKEN` if they are
   temporary — e.g. already produced by someone else's `sts:AssumeRole` call) — used directly.
   This identity needs `deployer_policy.json` itself, not just permission to assume a role.
3. Nothing set — boto3's default chain, same as everywhere else in this repository.

```bash
export AWS_REGION=eu-central-1 AWS_ACCOUNT_ID=123456789012
export ARTIFACT_BUCKET=my-ml-factory-artifacts APPROVED_DATA_BUCKET=my-approved-data
export DEFINITIONS_BUCKET=my-deploy-bucket
export DEPLOYER_ROLE_ARN=arn:aws:iam::123456789012:role/MlFactoryDeployer  # or the key pair above
make deploy-aws-nocli
```

## Running the control plane

The control plane is one container on the corporate server — the one permanently available host
in this architecture. It needs three things: Docker with the compose plugin, outbound HTTPS to
AWS (through the corporate proxy if that is how egress works), and credentials for
`MlFactoryBackendRole`.

```bash
git clone https://github.com/<your-org>/auto_eda_ml_pipeline.git
cd auto_eda_ml_pipeline
cp .env.example .env
cp config/secrets/config.sample.py config/secrets/config.py   # fill in the stack outputs
docker network create dev_network            # once per host, if it does not exist yet
docker compose up -d --build
# http://<server>:7570
```

For credentials, in order of preference:

1. **Mount an AWS profile.** If the server already has `~/.aws` configured with a profile whose
   role is `MlFactoryBackendRole`, mount it read-only into the container and leave the keys in
   `config/secrets/config.py` as `None`. Nothing to rotate in this repository.
2. **Static keys in `config/secrets/config.py`.** An IAM user with permission to assume
   `MlFactoryBackendRole` — the stack's trust policy allows any principal in the account, so
   grant that user `sts:AssumeRole` on the role ARN and nothing else. The file is gitignored;
   rotate on your normal schedule.

`GET /api/v1/health` reports which credential source is in effect — without ever returning a
credential — and lists any configuration that would fail at runtime, so a misconfigured
deployment says so up front instead of failing on the first experiment.

Put the server behind your normal corporate reverse proxy if it needs the company domain and TLS
certificate; the container itself serves plain HTTP on 7570.

`make deploy-aws` does not touch the corporate server, and the corporate server does not run any
AWS-side resource. The two steps are independent: `make deploy-aws` is per release, `docker
compose up` is per host.

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
