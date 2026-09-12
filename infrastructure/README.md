# Infrastructure

Everything AWS-side. No always-on ML compute: the only permanently running component is the
`docker compose` container (FastAPI control plane + built React app), wherever you choose to
run it — an EC2 instance is the natural AWS-native choice and is what this document walks
through, but the container is just a container; it runs anywhere with outbound access to AWS.

```
stepfunctions/   the two workflow definitions (ASL)
iam/             the three runtime role boundaries, plus the deployer's own permissions
cloudformation/  one stack: artifact bucket, KMS key, roles, an EC2 instance profile, workflows
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

Two different kinds of "permissions" are easy to conflate. **Runtime roles** are what the
platform itself assumes while it runs — you never use these credentials yourself. The
**deployer's permissions** are what your own AWS identity (an IAM user, an SSO role, a CI role)
needs to stand the platform up in the first place. They're deliberately separate: the deployer
can create and configure the stack without being able to act as the running platform, and vice
versa.

### Runtime roles (the platform's own credentials)

| Role | Trusted by | May do | May not do |
|---|---|---|---|
| `MlFactoryBackendRole` | `ec2.amazonaws.com` (as an instance profile) or `sts:AssumeRole` from the account | start and describe executions, read/write the artifact prefix (records included), read approved datasets, invoke the allow-listed Bedrock model | create SageMaker jobs, touch any other bucket |
| `MlFactoryWorkflowRole` | `states.amazonaws.com` | create/describe/stop `mlf-*` jobs, pass the job role to SageMaker, write `…/experiments/*/state/*` | read datasets or artifacts directly |
| `MlFactoryJobRole` | `sagemaker.amazonaws.com` | read the approved dataset prefix, read/write one experiment artifact prefix, record the terminal state, write logs | start further jobs, reach other buckets |

The policy documents in `iam/backend_role_policy.json`, `workflow_role_policy.json` and
`job_role_policy.json` are the reference version with `${PLACEHOLDER}` variables; the
CloudFormation template contains the deployed equivalents with real ARNs. You never assume
these roles by hand — `MlFactoryBackendRole` is what the EC2 instance running `docker compose`
is given via its instance profile (see below), and the other two are assumed by AWS services.

### Deployer permissions (your own AWS identity)

`iam/deployer_policy.json` is what to attach to whoever — or whatever CI role — runs
`make deploy-aws` and, separately, whoever launches the EC2 host. It covers exactly what that
requires: create/update the CloudFormation stack; create the three runtime roles and the
backend instance profile (`CAPABILITY_NAMED_IAM`) and pass them to the services that use them;
create the artifact bucket and its KMS key; push to ECR; upload the workflow definitions;
create the two state machines; and, for the person launching the host, run an EC2 instance with
the backend instance profile attached. It is not `AdministratorAccess`, and it is not one of
the three runtime roles above — attaching it to an operator does not let that operator act as
the running platform, only stand it up.

If your organization requires a first deploy under a broader policy (e.g. while validating the
resource list), `deployer_policy.json` is still the target to narrow down to afterward — every
action in it traces to a specific step below.

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

## Running `docker compose` on AWS

`docker compose` runs the control-plane container on a host; it is not itself an AWS service,
so it needs somewhere to run. The stack already creates what an EC2 host needs — an instance
profile carrying `MlFactoryBackendRole` — for exactly this: attach it and the container gets
S3/Step Functions/Bedrock credentials from the instance metadata service, with no keys to type
into `config/secrets/config.py` and nothing to rotate.

The mandate that ruled out ECS, Fargate, always-on ML compute and SageMaker endpoints did not
rule out EC2 for the one thing that must run continuously: the small control plane. A single
`t3.small`-class instance is enough — it validates, records and starts executions; it never
does the heavy compute.

```bash
# 1. Security group: inbound 7570 (or restrict to your corporate network / put an ALB in
#    front and open only to the ALB), inbound 22 for your own access, all outbound (S3, ECR,
#    Step Functions and Bedrock are reached over the public AWS endpoints).
aws ec2 create-security-group --group-name ml-factory-host \
  --description "ML Factory control plane" --vpc-id "$VPC_ID"
aws ec2 authorize-security-group-ingress --group-id "$SG_ID" \
  --protocol tcp --port 7570 --cidr "$YOUR_CIDR"
aws ec2 authorize-security-group-ingress --group-id "$SG_ID" \
  --protocol tcp --port 22 --cidr "$YOUR_CIDR"

# 2. Launch the instance with the stack's instance profile attached - this is what gives the
#    container credentials with nothing to configure.
aws ec2 run-instances \
  --image-id "$AMAZON_LINUX_2023_AMI_ID" \
  --instance-type t3.small \
  --iam-instance-profile Name=MlFactoryBackendInstanceProfile \
  --security-group-ids "$SG_ID" \
  --subnet-id "$SUBNET_ID" \
  --key-name "$YOUR_KEY_PAIR" \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ml-factory}]'

# 3. On the instance: install Docker + the compose plugin, clone the repo, configure, run.
sudo yum install -y docker git
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"          # re-login for this to take effect
DOCKER_CONFIG=${DOCKER_CONFIG:-$HOME/.docker}
mkdir -p "$DOCKER_CONFIG/cli-plugins"
curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
  -o "$DOCKER_CONFIG/cli-plugins/docker-compose"
chmod +x "$DOCKER_CONFIG/cli-plugins/docker-compose"

git clone https://github.com/<your-org>/auto_eda_ml_pipeline.git
cd auto_eda_ml_pipeline
cp .env.example .env
cp config/secrets/config.sample.py config/secrets/config.py   # fill in the stack outputs
docker network create dev_network
docker compose up -d --build
# http://<instance-public-or-private-ip>:7570
```

With the instance profile attached, leave `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` unset
in `config/secrets/config.py` — boto3's default credential chain finds the instance role
automatically, and `GET /api/v1/health` reports `"credential_source": "default AWS chain"` to
confirm it. Put the instance behind your corporate reverse proxy or an internal ALB if it needs
to be reachable under the company's normal domain and TLS certificate; the container itself
only serves plain HTTP on 7570.

`make deploy-aws` does not provision the EC2 host — it deploys the AWS-side resources the host
depends on. Provisioning the host is a one-time, infrastructure-as-code-of-your-choice step
(the `aws ec2` commands above, or your own Terraform/CDK/CloudFormation if you already manage
compute that way); nothing about the container assumes any particular way of launching it.

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
