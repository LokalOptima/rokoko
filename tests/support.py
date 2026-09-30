"""Small shared utilities for local GPU checks and reports (standard library only)."""
import contextlib
import hashlib
import io
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def command(args):
    p = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, timeout=30)
    return p.stdout.strip() if p.returncode == 0 else p.stderr.strip()

def provenance(binary=None):
    result = dict(date=command(['date', '-u', '+%FT%TZ']), commit=command(['git','rev-parse','HEAD']),
        dirty_patch_sha256=hashlib.sha256(command(['git','diff','HEAD']).encode()).hexdigest(),
        status=command(['git','status','--short']), gpu=command(['nvidia-smi',
        '--query-gpu=name,driver_version,memory.total,temperature.gpu,clocks.sm,utilization.gpu,power.draw',
        '--format=csv,noheader']))
    result['command_line']=__import__('sys').argv
    files=[]
    for folder in ('src','tests'):
        for path in sorted((ROOT/folder).rglob('*')):
            if path.is_file() and path.suffix in ('.cpp','.cu','.h','.py','.json','.tsv','.txt') and not any(x in path.parts for x in ('results','probe','models','__pycache__')):
                files.append((str(path.relative_to(ROOT)),sha256(path)))
    result['source_files']=dict(files)
    if binary: result.update(binary=str(binary), binary_sha256=sha256(binary))
    return result

def model_args(parser):
    cache = Path(os.environ.get('XDG_CACHE_HOME', str(Path.home()/'.cache'))) / 'rokoko'
    parser.add_argument('--models', type=Path, default=cache)
    parser.add_argument('--g2p', type=Path)
    parser.add_argument('--voices', type=Path)

def paths(args, binary):
    return dict(weights=args.models / ('weights.fp16.bin' if 'fp16' in Path(binary).name else 'weights.bin'),
                g2p=args.g2p or args.models/'g2p.bin', voices=args.voices or args.models/'voices')

def model_flags(args, binary):
    result=[]
    for key, value in paths(args, binary).items():
        if not value.exists(): raise FileNotFoundError(f'missing local {key}: {value}')
        result += ['--'+key, str(value.resolve())]
    return result

def idle_gpu():
    p = subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name', '--format=csv,noheader'],
                       capture_output=True, text=True, check=True)
    if p.stdout.strip(): raise RuntimeError('GPU has active compute processes: '+p.stdout.strip())

def request(base, endpoint, payload=None, raw=None):
    data = raw if raw is not None else json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None
    req = urllib.request.Request(base+endpoint, data=data,
        headers={'Content-Type':'application/json'} if data is not None else {})
    start=time.perf_counter()
    try: response=urllib.request.urlopen(req, timeout=180)
    except urllib.error.HTTPError as e: response=e
    with response:
        body=response.read()
        return response.status, dict(response.headers), body, (time.perf_counter()-start)*1000

def wav_info(data):
    with wave.open(io.BytesIO(data)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1,2,24000), 'WAV format'
        n=w.getnframes()
        samples=w.readframes(n)
        assert n > 0 and len(samples) == 2*n, 'empty or truncated WAV'
        assert len(data) == 44+2*n, 'RIFF/sample lengths'
        return dict(samples=n, audio_seconds=n/24000, sha256=hashlib.sha256(data).hexdigest())

@contextlib.contextmanager
def server(binary, args, log, extra=()):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
    base=f'http://127.0.0.1:{port}'
    log.parent.mkdir(parents=True, exist_ok=True)
    start=time.perf_counter()
    with log.open('wb') as f:
        process=subprocess.Popen([str(Path(binary).resolve()),'--serve',str(port),'--host','127.0.0.1',
                                  *model_flags(args,binary), *extra],stdout=f,stderr=f)
        try:
            for _ in range(600):
                if process.poll() is not None: raise RuntimeError(f'server exited {process.returncode}: {log.read_text()[-3000:]}')
                try:
                    status,_,_,_=request(base,'/health')
                    if status == 200: break
                except (OSError, urllib.error.URLError): pass
                time.sleep(.05)
            else: raise TimeoutError('server did not become healthy')
            yield base, (time.perf_counter()-start)*1000
        finally:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()

def save_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
