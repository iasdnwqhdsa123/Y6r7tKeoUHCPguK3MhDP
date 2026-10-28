from __future__ import annotations
import json
import os
import sys
ALWAYS = ('t2', 'td2', 'u_sfc', 'v_sfc', 'cloud', 'mslp')

def _missing_in(root: str, run_dir: str, days: dict, fields: list[str]) -> list[str]:
    out = []
    need = [f for f in fields if f != 'elevation' and (not f.startswith('q_'))]
    if not need:
        return []
    for day in days:
        folder = os.path.join(root, run_dir, day)
        if not os.path.isdir(folder):
            out.append(f'{run_dir}/{day}: no folder')
            continue
        have = {n[:-5] for n in os.listdir(folder) if n.endswith('.pack')} | {n for n in os.listdir(folder) if os.path.isdir(os.path.join(folder, n))}
        gone = [f for f in ALWAYS if f in need and f not in have]
        if gone:
            out.append(f'{run_dir}/{day}: no {', '.join(gone)}')
    return out

def check_manifest(root: str, rel: str) -> list[str]:
    path = os.path.join(root, rel, 'latest.json')
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        m = json.load(f)
    days, run = (m.get('days') or {}, m.get('runId'))
    if not days or not run:
        return []
    base = os.path.join(rel, run)
    fields = [k for k, v in (m.get('fields') or {}).items() if v.get('tiles')]
    out = _missing_in(root, base, days, fields)
    if 'elevation' in fields and (not any((os.path.exists(os.path.join(root, base, n)) for n in ('elevation.pack', 'elevation')))):
        out.append(f'{base}: no terrain')
    for lv in m.get('levels') or []:
        lf = [k for k, v in (lv.get('tiles') or {}).items() if v]
        out += _missing_in(root, os.path.join(base, lv['path']), days, lf)
    return out

def main() -> int:
    root, model = (sys.argv[1], sys.argv[2])
    gone = []
    for rel in (model, os.path.join('free', model)):
        gone += [f'{rel}: {g}' for g in check_manifest(root, rel)]
    if gone:
        print(f'{model}: output not whole ({len(gone)} problems):')
        for g in gone[:20]:
            print('  ' + g)
        return 1
    print(f'{model}: output whole')
    return 0
if __name__ == '__main__':
    sys.exit(main())
