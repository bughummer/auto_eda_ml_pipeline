import os, json, logging, boto3
from botocore.exceptions import ClientError
log = logging.getLogger(); log.setLevel(logging.INFO)
POLICY_ARN = os.environ['DENY_POLICY_ARN']
PRINCIPALS = [p.strip() for p in os.environ.get('PRINCIPALS','').split(',') if p.strip()]
AUTO_KILL  = os.environ.get('AUTO_KILL','false').lower() == 'true'
iam = boto3.client('iam')

def attach(principal):
    for fn, kind in [(iam.attach_user_policy, 'user'),
                     (iam.attach_role_policy, 'role')]:
        try:
            kw = {f'{kind.title()}Name': principal, 'PolicyArn': POLICY_ARN}
            fn(**kw)
            log.info('attached deny to %s %s', kind, principal)
            return kind
        except ClientError as e:
            if e.response['Error']['Code'] in ('NoSuchEntity','NoSuchEntityException'):
                continue
            log.error('attach to %s %s failed: %s', kind, principal, e)
            return None
    log.warning('principal not found as user or role: %s', principal)
    return None

def handler(event, _ctx):
    log.info('event=%s', json.dumps(event)[:1500])
    if not PRINCIPALS:
        return {'status':'no_principals'}
    if not AUTO_KILL:
        log.info('[dry-run] AUTO_KILL=false; would attach %s to: %s', POLICY_ARN, PRINCIPALS)
        return {'status':'dry_run','principals':PRINCIPALS}
    return {'status':'ok','results':{p: attach(p) for p in PRINCIPALS}}
