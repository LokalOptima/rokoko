"""Run a copied executable alone, with outbound connect calls denied and file access traced."""
import argparse
import hashlib
import json
import mmap
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
from support import ROOT, server, request, wav_info, embedded_identity, save_report


def verify_embedded_bytes(binary, info):
    # Independently hash the linked ELF bytes, rather than trusting --build-info.
    symbols={}
    payloads=[]
    for line in subprocess.check_output(['nm','-S','--defined-only',str(binary)],text=True).splitlines():
        fields=line.split()
        if len(fields)==4 and fields[3].startswith('rokoko_') and fields[3].endswith('_start'):
            symbols[fields[3]]=(int(fields[0],16),int(fields[1],16))
    with binary.open('rb') as f:
        header=f.read(64)
        assert header[:6]==b'\x7fELF\x02\x01','expected Linux ELF64 little-endian binary'
        offset=struct.unpack_from('<Q',header,32)[0]
        entry_size,count=struct.unpack_from('<HH',header,54)
        segments=[]
        for i in range(count):
            f.seek(offset+i*entry_size)
            segments.append(struct.unpack('<IIQQQQQQ',f.read(56)))
        for name,meta in info['files'].items():
            kind='weights' if name.startswith('weights') else 'g2p' if name=='g2p.bin' else 'voice'
            address,size=symbols['rokoko_'+kind+'_start']
            assert size==meta['size'] and address%4096==0,(name,'size/alignment')
            matches=[p for p in segments if p[0]==1 and p[3]<=address and address+size<=p[3]+p[5]]
            assert len(matches)==1 and not matches[0][1]&2,(name,'writable or unmapped asset')
            segment=matches[0];position=segment[2]+address-segment[3]
            payloads.append((name,position,size))
            f.seek(position);h=hashlib.sha256()
            remaining=size
            while remaining:
                block=f.read(min(1024*1024,remaining));assert block
                h.update(block);remaining-=len(block)
            assert h.hexdigest()==meta['sha256'],(name,'embedded bytes differ from approved asset')
    assert set(symbols)=={'rokoko_'+kind+'_start' for kind in ('weights','g2p','voice','info')}
    # Check actual payload duplication: static inference libraries can legitimately
    # grow the executable, so a fixed allowance for non-asset bytes is misleading.
    with binary.open('rb') as f, mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as data:
        for name,position,size in payloads:
            with memoryview(data)[position:position+size] as payload:
                assert data.find(payload)==position,(name,'earlier duplicate asset payload')
                assert data.find(payload,position+1)==-1,(name,'duplicate asset payload')


def trace_command(trace):
    return [shutil.which('strace'),'-D','-f','-qq','-e','trace=%file,%network,%process',
            '-e','inject=connect:error=ENETUNREACH','-o',str(trace)]


def check_trace(path):
    trace=path.read_text()
    for name in ('weights.bin','weights.fp16.bin','g2p.bin','af_heart.bin','.cache/rokoko'):
        assert name not in trace, f'external model access in {path}: {name}'
    for line in trace.splitlines():
        if 'connect(' in line:
            assert 'ENETUNREACH' in line, f'unblocked connection: {line}'


def check_binary(original):
    with tempfile.TemporaryDirectory(prefix='rokoko-bundle-') as directory:
        d=Path(directory);binary=d/'tts';shutil.copy2(original,binary)
        # These regular files make any HOME/XDG cache directory creation fail.
        (d/'home').touch();(d/'cache').touch();(d/'tmp').mkdir();(d/'empty-path').mkdir()
        env=dict(os.environ,HOME=str(d/'home'),XDG_CACHE_HOME=str(d/'cache'),TMPDIR=str(d/'tmp'),PATH=str(d/'empty-path'),CUDA_CACHE_DISABLE='1')
        info=embedded_identity(binary)
        assert info['precision']=='fp16'
        verify_embedded_bytes(binary,info)
        output=d/'out.wav';trace=d/'cli.trace'
        subprocess.run([*trace_command(trace),str(binary),'Hello from a standalone executable.','-o',str(output)],
                       env=env,cwd=d,check=True,capture_output=True,timeout=120)
        result=dict(build=info,cli=wav_info(output.read_bytes()))
        check_trace(trace)
        trace=d/'server.trace'
        with server(binary,None,d/'server.log',env=env,cwd=d,prefix=trace_command(trace)) as (base,startup):
            status,_,data,_=request(base,'/synthesize',{'text':'Bundled HTTP audio.'})
            assert status==200;result['http']=wav_info(data)
            status,_,data,_=request(base,'/synthesize/stream',{'text':'Bundled streaming audio.'})
            assert status==200 and len(data)>0 and len(data)%4==0
            import math
            assert all(math.isfinite(v[0]) for v in struct.iter_unpack('<f',data))
            result['stream_samples']=len(data)//4
        check_trace(trace)
        assert list((d/'tmp').iterdir())==[],'runtime extracted temporary assets'
        assert {p.name for p in d.iterdir()}=={'tts','home','cache','tmp','empty-path','out.wav','cli.trace','server.trace','server.log'}
        result['isolation']='Copied executable; no model file accesses; all connect syscalls denied; HOME/cache unavailable; empty PATH; no extracted assets.'
        return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary',type=Path,action='append')
    args=ap.parse_args()
    if not shutil.which('strace'): raise RuntimeError('strace is required for the binary-only distribution check')
    report={str(p):check_binary(p) for p in (args.binary or [ROOT/'rokoko'])}
    save_report(ROOT/'tests/results/bundle.json',report)
    print('PASS standalone CLI, HTTP and streaming without external model files, runtime downloads or extracted assets.')

if __name__=='__main__':main()
