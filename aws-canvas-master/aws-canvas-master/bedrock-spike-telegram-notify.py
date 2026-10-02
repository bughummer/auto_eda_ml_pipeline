import os, json, logging, re
from urllib.request import Request, urlopen
from urllib.parse import urlencode
log = logging.getLogger(); log.setLevel(logging.INFO)
TOKEN = os.environ['TG_BOT_TOKEN']
CHAT_IDS = [c.strip() for c in os.environ['TG_CHAT_IDS'].split(',') if c.strip()]
PRINCIPALS = [p.strip() for p in os.environ.get('BEDROCK_PRINCIPALS','').split(',') if p.strip()]
AUTO_KILL = os.environ.get('AUTO_KILL','false').lower() == 'true'
ACCOUNT_LABEL = os.environ.get('ACCOUNT_LABEL','').strip()

MD_SPECIAL = re.compile(r'([_*`\[])')
def esc(s):
    return MD_SPECIAL.sub(r'\\\1', str(s))

def fmt_num(n):
    try:
        return f"{float(n):,.0f}"
    except Exception:
        return str(n)

def parse_reason(reason):
    # "Threshold Crossed: 2 out of the last 2 datapoints [93646.0 (25/05/26 12:43:00), 51050.0 (...)] were greater than the threshold (5000.0) ..."
    dps = re.findall(r'([\d.]+)\s*\(([^)]+)\)', reason or '')
    thr = re.search(r'threshold\s*\(([\d.]+)\)', reason or '', re.I)
    return dps, (thr.group(1) if thr else None)

def format_cw_alarm(msg):
    name = msg.get('AlarmName', 'unknown')
    state = msg.get('NewStateValue', '')
    region = msg.get('Region', '')
    account = msg.get('AWSAccountId', '')
    ts = msg.get('StateChangeTime', '')
    trig = msg.get('Trigger', {}) or {}
    metric = trig.get('MetricName', '')
    ns = trig.get('Namespace', '')
    stat = trig.get('Statistic', '')
    period = trig.get('Period', '')
    threshold = trig.get('Threshold', '')
    reason = msg.get('NewStateReason', '')
    dps, _ = parse_reason(reason)

    icon = '\U0001f6a8' if state == 'ALARM' else ('\u2705' if state == 'OK' else '\u2139\ufe0f')
    header = f"Bedrock alert: {esc(name)}"
    if ACCOUNT_LABEL:
        header = f"[{esc(ACCOUNT_LABEL)}] {header}"
    acct_line = (f"Account: *{esc(ACCOUNT_LABEL)}* (`{esc(account)}`) \u2022 Region: {esc(region)}"
                 if ACCOUNT_LABEL
                 else f"Account: `{esc(account)}` \u2022 Region: {esc(region)}")
    lines = [
        f"{icon} *{header}*",
        f"State: *{esc(state)}*",
        f"Metric: `{esc(ns)}/{esc(metric)}` ({esc(stat)}, {esc(period)}s)",
        f"Threshold: *{fmt_num(threshold)}*",
    ]
    if dps:
        recent = ", ".join(f"{fmt_num(v)} @ {esc(t)}" for v, t in dps[:3])
        lines.append(f"Recent datapoints: {recent}")
    lines.append(acct_line)
    if ts:
        lines.append(f"At: {esc(ts)}")
    if PRINCIPALS:
        mode = "auto-deny attached" if (state == 'ALARM' and AUTO_KILL) else "kill list (notify-only)"
        lines.append(f"Principals \u2014 {esc(mode)}: {esc(', '.join(PRINCIPALS))}")
    return "\n".join(lines)

def format_budget(msg):
    # AWS Budgets sends a human-readable string, not JSON. Just trim.
    first = msg.split('\n', 1)[0][:500]
    prefix = f"[{esc(ACCOUNT_LABEL)}] " if ACCOUNT_LABEL else ""
    return f"\U0001f4b0 *{prefix}Bedrock budget alert*\n{esc(first)}"

def format_cost_anomaly(msg):
    acct = msg.get('accountId', '')
    impact = msg.get('impact', {}) or {}
    total = impact.get('totalImpact', impact.get('totalActualSpend', ''))
    service = ''
    for d in (msg.get('rootCauses') or []):
        service = d.get('service', '') or service
    start = msg.get('anomalyStartDate', '')
    end = msg.get('anomalyEndDate', '')
    prefix = f"[{esc(ACCOUNT_LABEL)}] " if ACCOUNT_LABEL else ""
    acct_line = (f"Account: *{esc(ACCOUNT_LABEL)}* (`{esc(acct)}`)"
                 if ACCOUNT_LABEL else f"Account: `{esc(acct)}`")
    return ("\U0001f4c8 *" + prefix + "Cost anomaly detected*\n"
            f"Service: *{esc(service or 'Bedrock')}*\n"
            f"Impact: *${fmt_num(total)}*\n"
            f"Window: {esc(start)} \u2192 {esc(end)}\n"
            f"{acct_line}")

def build_message(subject, raw):
    try:
        msg = json.loads(raw)
    except Exception:
        return f"\U0001f6a8 *{esc(subject or 'Bedrock alert')}*\n{esc(raw[:1500])}"
    if isinstance(msg, dict) and 'AlarmName' in msg and 'NewStateValue' in msg:
        return format_cw_alarm(msg)
    if isinstance(msg, dict) and 'anomalyId' in msg:
        return format_cost_anomaly(msg)
    return f"\U0001f6a8 *{esc(subject or 'Bedrock alert')}*\n```\n{json.dumps(msg, indent=2)[:3500]}\n```"

def send_telegram(chat_id, text):
    params = urlencode({
        'chat_id': chat_id,
        'text': text[:4096],
        'parse_mode': 'Markdown',
        'disable_web_page_preview': 'true',
    }).encode()
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    req = Request(url, data=params, method='POST',
                  headers={'Content-Type': 'application/x-www-form-urlencoded'})
    try:
        resp = urlopen(req, timeout=5)
        log.info('Telegram chat_id=%s response: %s', chat_id, resp.read().decode())
    except Exception as e:
        log.error('Failed to send to chat_id=%s: %s', chat_id, e)

def handler(event, _ctx):
    for record in event.get('Records', []):
        sns = record.get('Sns', {})
        subject = sns.get('Subject') or 'Bedrock alert'
        raw = sns.get('Message', '') or ''
        if subject and 'Budget' in subject and not raw.lstrip().startswith('{'):
            text = format_budget(raw)
        else:
            text = build_message(subject, raw)
        for chat_id in CHAT_IDS:
            send_telegram(chat_id, text)
