"""Fetch checksum-pinned official reference files; verify supplied runtime artifacts locally."""
import argparse
import json
import urllib.request
from pathlib import Path
from support import ROOT, sha256

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--official',type=Path,default=ROOT/'tests/models/Kokoro-82M')
    ap.add_argument('--models',type=Path,required=True,help='directory containing the approved runtime files')
    args=ap.parse_args();manifest=json.loads((ROOT/'tests/fixtures/artifacts.json').read_text())
    for name,meta in manifest['files'].items():
        path=args.models/name
        if not path.exists() or path.stat().st_size!=meta['size'] or sha256(path)!=meta['sha256']:
            raise RuntimeError('missing or incorrect runtime artifact: '+str(path)+'; preparation never substitutes a different model')
    for name,meta in manifest['official'].items():
        path=args.official/name
        if path.exists():
            if path.stat().st_size!=meta['size'] or sha256(path)!=meta['sha256']: raise RuntimeError('existing official file differs: '+str(path))
            continue
        path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.download')
        url=f"https://huggingface.co/{manifest['source']}/resolve/{manifest['source_revision']}/{name}"
        try:
            urllib.request.urlretrieve(url,temporary)
            if temporary.stat().st_size!=meta['size'] or sha256(temporary)!=meta['sha256']:raise RuntimeError('download checksum mismatch: '+name)
            temporary.replace(path)
        finally:temporary.unlink(missing_ok=True)
    print('Prepared pinned official references and verified local runtime artifacts.')
if __name__=='__main__':main()
