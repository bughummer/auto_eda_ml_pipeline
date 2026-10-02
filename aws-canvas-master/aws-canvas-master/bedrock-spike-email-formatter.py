import os, json, logging, re, boto3
log = logging.getLogger(); log.setLevel(logging.INFO)
OUTPUT_TOPIC = os.environ['OUTPUT_TOPIC_ARN']
PRINCIPALS = [p.strip() for p in os.environ.get('BEDROCK_PRINCIPALS','').split(',') if p.strip()]
AUTO_KILL = os.environ.get('AUTO_KILL','false').lower() == 'true'
ACCOUNT_LABEL = os.environ.get('ACCOUNT_LABEL','').strip()
sns = boto3.client('sns')

def fmt_num(n):
    try: return f"{float(n):,.0f}"
    except Exception: return str(n)

def parse_reason(reason):
    dps = re.findall(r'([\d.]+)\s*\(([^)]+)\)', reason or '')
    return dps

def cw_alarm_body(msg):
    name = msg.get('AlarmName','unknown')
    state = msg.get('NewStateValue','')
    region = msg.get('Region','')
    account = msg.get('AWSAccountId','')
    ts = msg.get('StateChangeTime','')
    trig = msg.get('Trigger',{}) or {}
    lines = [
        f"Alarm:      {name}",
        f"State:      {state}",
        f"Metric:     {trig.get('Namespace','')}/{trig.get('MetricName','')} ({trig.get('Statistic','')}, {trig.get('Period','')}s)",
        f"Threshold:  {fmt_num(trig.get('Threshold',''))}",
    ]
    dps = parse_reason(msg.get('NewStateReason',''))
    if dps:
        lines.append("Recent datapoints:")
        for v, t in dps[:5]:
            lines.append(f"  - {fmt_num(v)} @ {t}")
    acct_disp = f"{ACCOUNT_LABEL} ({account})" if ACCOUNT_LABEL else account
    lines += [f"Account:    {acct_disp}", f"Region:     {region}", f"At:         {ts}"]
    if PRINCIPALS:
        mode = "auto-deny attached" if (state == 'ALARM' and AUTO_KILL) else "kill list (notify-only)"
        lines.append(f"Principals ({mode}): {', '.join(PRINCIPALS)}")
    subject = f"[{ACCOUNT_LABEL or account}] Bedrock {state}: {name}"
    return subject, "\n".join(lines)

def cost_anomaly_body(msg):
    acct = msg.get('accountId','')
    impact = msg.get('impact',{}) or {}
    total = impact.get('totalImpact', impact.get('totalActualSpend',''))
    service = next((d.get('service','') for d in (msg.get('rootCauses') or []) if d.get('service')), 'Bedrock')
    start = msg.get('anomalyStartDate','')
    end = msg.get('anomalyEndDate','')
    acct_disp = f"{ACCOUNT_LABEL} ({acct})" if ACCOUNT_LABEL else acct
    body = (f"Service:  {service}\n"
            f"Impact:   ${fmt_num(total)}\n"
            f"Window:   {start} -> {end}\n"
            f"Account:  {acct_disp}")
    subject = f"[{ACCOUNT_LABEL or acct}] Bedrock cost anomaly: ${fmt_num(total)}"
    return subject, body

def budget_body(raw):
    first = (raw or '').split('\n', 1)[0][:500]
    subject = f"[{ACCOUNT_LABEL}] Bedrock budget alert" if ACCOUNT_LABEL else "Bedrock budget alert"
    return subject, first

def handler(event, _ctx):
    for record in event.get('Records', []):
        sns_msg = record.get('Sns', {})
        raw = sns_msg.get('Message','') or ''
        in_subject = sns_msg.get('Subject','') or ''
        try:
            msg = json.loads(raw)
        except Exception:
            msg = None
        if isinstance(msg, dict) and 'AlarmName' in msg and 'NewStateValue' in msg:
            subject, body = cw_alarm_body(msg)
        elif isinstance(msg, dict) and 'anomalyId' in msg:
            subject, body = cost_anomaly_body(msg)
        elif 'Budget' in in_subject:
            subject, body = budget_body(raw)
        else:
            subject = f"[{ACCOUNT_LABEL}] {in_subject}".strip() or in_subject or 'Bedrock alert'
            body = raw[:4000]
        try:
            sns.publish(TopicArn=OUTPUT_TOPIC, Subject=subject[:99], Message=body)
        except Exception as e:
            log.error('publish failed: %s', e)
