#!/usr/bin/env python3
"""Verify/download pinned build inputs and generate GNU assembler resource wrappers."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'assets/manifest.json'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(path, meta):
    if path.stat().st_size != meta['size'] or digest(path) != meta['sha256']:
        raise ValueError(f'Asset checksum mismatch: {path}; remove the invalid build copy or supply the approved file')


def prepare_one(name, meta, directory, source=None, offline=False):
    destination = directory / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Share downloads safely between simultaneous builds.
    with (directory / '.assets.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            verify(destination, meta)
            return destination
        fd, temporary = tempfile.mkstemp(prefix=destination.name + '.', dir=destination.parent)
        os.close(fd)
        temporary = Path(temporary)
        try:
            if source is not None:
                local = source / name
                verify(local, meta)
                shutil.copyfile(local, temporary)
            elif offline:
                raise FileNotFoundError(f'Missing offline build asset: {destination}')
            else:
                print(f'Downloading {name} into {directory}', flush=True)
                request = urllib.request.Request(meta['url'], headers={'User-Agent': 'rokoko-build'})
                try:
                    with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as f:
                        shutil.copyfileobj(response, f)
                except urllib.error.URLError as error:
                    raise RuntimeError(f'Cannot download {name}: {error}. Supply approved files with ASSET_SOURCE (Make) or ROKOKO_ASSET_SOURCE (CMake).') from error
            verify(temporary, meta)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


def write_changed(path, text):
    if path.exists() and path.read_text() == text:
        return
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(text)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def assembly(resources):
    lines = ['.section .rodata.rokoko,"a",@progbits']
    for symbol, path in resources:
        # JSON quoting also escapes GNU assembler strings on the supported Linux target.
        start = 'rokoko_' + symbol + '_start'
        end = 'rokoko_' + symbol + '_end'
        lines += ['.balign 4096', f'.global {start}, {end}', f'.hidden {start}, {end}',
                  f'.type {start}, @object', start + ':', '.incbin ' + json.dumps(str(path.resolve())),
                  end + ':', f'.size {start}, {end} - {start}']
    lines += ['.section .note.GNU-stack,"",@progbits', '']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--directory', type=Path, required=True)
    ap.add_argument('--source', type=Path)
    ap.add_argument('--offline', action='store_true')
    args = ap.parse_args()
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    files = json.loads(MANIFEST.read_text())['files']
    weights = 'weights.fp16.bin'
    names = [('weights', weights), ('g2p', 'g2p.bin'), ('voice', 'voices/af_heart.bin')]
    resources = [(symbol, prepare_one(name, files[name], directory, args.source, args.offline)) for symbol, name in names]
    info = dict(precision='fp16', voice='af_heart', files={name: {k: files[name][k] for k in ('size', 'sha256')} for _, name in names})
    info_path = directory / 'info.json'
    write_changed(info_path, json.dumps(info, indent=2) + '\n')
    resources.append(('info', info_path))
    write_changed(directory / 'embedded.S', assembly(resources))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error))
