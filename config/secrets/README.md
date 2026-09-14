# config/secrets

Two files, for two different times.

| File | Read by | When |
|---|---|---|
| `config.py` | the running platform | continuously, on the corporate server |
| `deploy.py` | `scripts/deploy_aws.py` / `deploy_aws.sh` | once, when you run `make deploy-aws*` |

Both samples are tracked and carry no real values. Copy each once:

```bash
cp config/secrets/config.sample.py config/secrets/config.py
cp config/secrets/deploy.sample.py config/secrets/deploy.py
```

Neither real file is ever committed — both are in `.gitignore`, so `git status` will not offer
them and `git add .` will not pick them up.

Values set in either file are overridden by an environment variable of the same name (with the
`ML_FACTORY_` prefix for `config.py`, no prefix for `deploy.py`), so a deployment or a single
invocation can change one value without editing the file. Anything left as `None` falls through
to the next layer.

Prefer a mounted `~/.aws` profile, or the deploy identity's own `DEPLOYER_ROLE_ARN` /
`AWS_ROLE_ARN`, over static access keys in either file. `GET /api/v1/health` reports the
running platform's credential source without ever returning a credential; `deploy_aws.py`
prints its own the same way, as it runs.
