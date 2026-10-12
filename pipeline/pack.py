import os
import shutil
import struct
import numpy as np
MAGIC = b'APK1'
HEAD = 12

def write_pack(path, blobs, ntx, nty):
    if len(blobs) != ntx * nty:
        raise ValueError(f'{len(blobs)} entries for a {ntx}×{nty} pack')
    ends = np.cumsum([len(b) for b in blobs], dtype=np.uint64) if blobs else np.zeros(0, np.uint64)
    if len(ends) and ends[-1] >= 2 ** 32:
        raise ValueError('pack over 4 GB')
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(MAGIC + struct.pack('<II', ntx, nty))
        f.write(ends.astype('<u4').tobytes())
        for b in blobs:
            f.write(b)
    os.replace(tmp, path)

def read_index(path):
    with open(path, 'rb') as f:
        head = f.read(HEAD)
        if head[:4] != MAGIC:
            raise ValueError(f'{path}: not a tile pack')
        ntx, nty = struct.unpack('<II', head[4:])
        ends = np.frombuffer(f.read(4 * ntx * nty), '<u4').astype(np.int64)
    return (ntx, nty, ends, HEAD + 4 * ntx * nty)

def keys_in(path):
    ntx, _, ends, _ = read_index(path)
    sizes = np.diff(ends, prepend=0)
    return {f'{i % ntx}_{i // ntx}' for i in np.nonzero(sizes)[0]}

def read_entry(path, i):
    _, _, ends, start = read_index(path)
    a = int(ends[i - 1]) if i else 0
    with open(path, 'rb') as f:
        f.seek(start + a)
        return f.read(int(ends[i]) - a)

def read_tile(path, key):
    ntx, nty, _, _ = read_index(path)
    tx, ty = (int(v) for v in key.split('_'))
    return read_entry(path, ty * ntx + tx) if tx < ntx and ty < nty else b''

def pack_folder(folder, keys, keep=False):
    blobs = []
    for k in keys:
        p = os.path.join(folder, f'{k}.png')
        if os.path.exists(p):
            with open(p, 'rb') as f:
                blobs.append(f.read())
        else:
            blobs.append(b'')
    write_pack(folder + '.pack', blobs, len(keys), 1)
    if not keep:
        shutil.rmtree(folder, ignore_errors=True)
