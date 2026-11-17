import argparse
import hashlib
import json
import os
import sys
import time
SKIP_DIRS = {'_hours', '_quarters'}

def files(root):
    out = []
    for d, dirs, names in os.walk(root):
        dirs[:] = sorted((x for x in dirs if not x.startswith('.') and x not in SKIP_DIRS and (not x.endswith('.building'))))
        for n in names:
            if n.startswith('.') or n.endswith('.tmp'):
                continue
            p = os.path.join(d, n)
            if os.path.isfile(p):
                out.append(os.path.relpath(p, root).replace(os.sep, '/'))
    return sorted(out)

def fingerprint(entries):
    h = hashlib.sha256()
    for rel, n in entries:
        h.update(f'{rel}\t{n}\n'.encode())
    return h.hexdigest()[:32]

def index(root, bundle, id_, now=None):
    entries = [(rel, os.path.getsize(os.path.join(root, rel))) for rel in files(root)]
    out, off = ({}, 0)
    for rel, n in entries:
        out[rel] = [off, n]
        off += n
    return {'v': 1, 'bundle': bundle, 'id': id_, 'time': int((now or time.time()) * 1000), 'size': off, 'fp': fingerprint(entries), 'files': out}

def cat(root, idx, out):
    sent = 0
    for rel, (off, n) in sorted(idx['files'].items(), key=lambda kv: kv[1][0]):
        if off != sent:
            raise SystemExit(f'bundle: {rel}: at {off} in the index, {sent} in the bundle')
        with open(os.path.join(root, rel), 'rb') as f:
            left = n
            while left:
                b = f.read(min(left, 1 << 20))
                if not b:
                    raise SystemExit(f'bundle: {rel}: shorter than the index says ({n})')
                out.write(b)
                left -= len(b)
            if f.read(1):
                raise SystemExit(f'bundle: {rel}: longer than the index says ({n})')
        sent += n
    if sent != idx['size']:
        raise SystemExit(f'bundle: {sent} bytes sent, the index says {idx['size']}')

def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('index')
    s.add_argument('dir')
    s.add_argument('--bundle', required=True)
    s.add_argument('--id', required=True)
    s = sub.add_parser('cat')
    s.add_argument('dir')
    s.add_argument('index')
    s = sub.add_parser('fp')
    s.add_argument('dir')
    a = ap.parse_args(argv)
    if a.cmd == 'index':
        json.dump(index(a.dir, a.bundle, a.id), sys.stdout, separators=(',', ':'))
    elif a.cmd == 'cat':
        with open(a.index) as f:
            idx = json.load(f)
        cat(a.dir, idx, sys.stdout.buffer)
        sys.stdout.buffer.flush()
    else:
        print(fingerprint([(rel, os.path.getsize(os.path.join(a.dir, rel))) for rel in files(a.dir)]))
if __name__ == '__main__':
    main()
