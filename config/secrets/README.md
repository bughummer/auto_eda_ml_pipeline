# config/secrets

`config.sample.py` is tracked and carries no real values. Copy it once per deployment:

```bash
cp config/secrets/config.sample.py config/secrets/config.py
```

`config/secrets/config.py` is in `.gitignore` and must never be committed. `git status` will
not offer it, and `git add .` will not pick it up.

Values set here are overridden by `ML_FACTORY_*` environment variables and by `.env`, so a
deployment can change one value without editing the file. Anything left as `None` falls
through to the next layer.

Prefer an instance role over static access keys. `GET /api/v1/health` reports which credential
source is in effect, without ever returning a credential.
