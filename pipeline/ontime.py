from __future__ import annotations
import json
import os
import urllib.request
from datetime import datetime, timedelta, timezone
TITLE = 'Forecast data (on time)'

def last_slot(cron: str, now: datetime) -> datetime:
    minute, hours = cron.split()[:2]
    cands = [now.replace(hour=int(h), minute=int(minute), second=0, microsecond=0) - timedelta(days=d) for d in (0, 1) for h in hours.split(',')]
    return max((t for t in cands if t <= now))

def stand_down(runs: list[dict], cron: str, now: datetime) -> bool:
    slot = last_slot(cron, now) - timedelta(minutes=10)
    return any((r.get('display_title') == TITLE and r.get('conclusion') not in ('cancelled', 'skipped') and (datetime.fromisoformat(r['created_at'].replace('Z', '+00:00')) >= slot) for r in runs))

def main():
    cron, repo, token = (os.environ.get('CRON', ''), os.environ.get('GITHUB_REPOSITORY', ''), os.environ.get('GH_TOKEN', ''))
    ref = os.environ.get('GITHUB_WORKFLOW_REF', '')
    wf = ref.split('/.github/workflows/', 1)[1].split('@', 1)[0] if '/.github/workflows/' in ref else 'forecast-data.yml'
    skip = False
    try:
        req = urllib.request.Request(f'https://api.github.com/repos/{repo}/actions/workflows/{wf}/runs?event=workflow_dispatch&per_page=20', headers={'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json'})
        with urllib.request.urlopen(req, timeout=20) as r:
            runs = json.load(r)['workflow_runs']
        skip = stand_down(runs, cron, datetime.now(timezone.utc))
        print(f'cron {cron!r}: {('the on-time trigger already ran this slot, standing down' if skip else 'no on-time run this slot, building')}')
    except Exception as e:
        print(f"couldn't check recent runs ({e}): building")
    out = os.environ.get('GITHUB_OUTPUT')
    if out:
        with open(out, 'a') as f:
            if cron:
                f.write(f'slot={last_slot(cron, datetime.now(timezone.utc)):%H:%M}\n')
            if skip:
                f.write('skip=true\n')
if __name__ == '__main__':
    main()
