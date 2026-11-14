import gzip
import os
import re
import sys
import time
import urllib.parse
import urllib.request
JMA_FEED = 'https://www.data.jma.go.jp/developer/xml/feed/regular_l.xml'
DWD_DIR = 'https://opendata.dwd.de/weather/charts/analysis/'
JMA_KINDS = ('VZSA60', 'VZSA50')
DWD_FILE = re.compile('Z__C_EDZW_\\d{14}_tka01,ana_bwkman_dwd(?:na|c)_O_000000_000000_(\\d{12})_WV12\\.png')

def get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'fronts-labels'}), timeout=60) as r:
                return r.read()
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))

def jma_wanted(feed):
    out = {}
    for url in re.findall('href="(https://www\\.data\\.jma\\.go\\.jp/developer/xml/data/[^"]+\\.xml)"', feed):
        name = url.rsplit('/', 1)[1]
        if any((f'_{k}_' in name for k in JMA_KINDS)) and re.match('\\d{14}_', name):
            out[url] = f'jma/{name[:6]}/{name}.gz'
    return sorted(out.items(), key=lambda kv: kv[1])

def dwd_wanted(listing):
    out = {}
    for href in re.findall('href="([^"]+)"', listing):
        name = urllib.parse.unquote(href)
        m = DWD_FILE.fullmatch(name)
        if m:
            out[DWD_DIR + href] = f'dwd/{m.group(1)[:6]}/{name}'
    return sorted(out.items(), key=lambda kv: kv[1])

def kept(listing):
    return {ln.split()[-1].split('fronts-labels/', 1)[-1] for ln in listing.splitlines() if ln.strip()}

def main(out, have_file=None):
    have = kept(open(have_file).read()) if have_file and os.path.exists(have_file) else set()
    n = failed = 0
    for source, listing_url, wanted in (('JMA', JMA_FEED, jma_wanted), ('DWD', DWD_DIR, dwd_wanted)):
        try:
            todo = [(u, k) for u, k in wanted(get(listing_url).decode('utf-8', 'replace')) if k not in have]
        except OSError as e:
            print(f'{source}: its listing failed: {e!r}')
            failed += 1
            continue
        for url, key in todo:
            try:
                body = get(url)
            except OSError as e:
                print(f'{source}: {key} failed: {e!r}')
                failed += 1
                continue
            path = os.path.join(out, key)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                f.write(gzip.compress(body, 9) if key.endswith('.gz') else body)
            n += 1
        print(f'{source}: {len(todo)} new')
    print(f'{n} files written, {failed} failed')
    return 0 if n or not failed else 1
if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:3]))
