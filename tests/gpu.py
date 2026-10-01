"""Bounded local artifact, CLI, library and HTTP regression checks. No downloads."""
import argparse
import hashlib
import http.client
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
from support import ROOT, model_args, paths, embedded_identity, server, request, wav_info, save_report, provenance, sha256, idle_gpu

VOICES=('af_heart',)
def verify_identity(args):
    manifest=json.loads((ROOT/'tests/fixtures/artifacts.json').read_text())
    def verify(name,path):
        expected=manifest['files'][name]
        assert path.stat().st_size==expected['size'] and sha256(path)==expected['sha256'], 'artifact identity: '+name
    for name in manifest['files']:
        path=(args.g2p if name=='g2p.bin' and args.g2p else (args.voices/Path(name).name if name.startswith('voices/') and args.voices else args.models/name))
        verify(name,path)
    with tempfile.TemporaryDirectory() as d:
        bad=Path(d)/'g2p.bin'; data=bytearray((args.g2p or args.models/'g2p.bin').read_bytes());data[-1]^=1;bad.write_bytes(data)
        try: verify('g2p.bin',bad)
        except AssertionError: pass
        else: raise AssertionError('corrupt identity control escaped')
        v8=args.models/'g2p_v8.bin'
        if v8.exists():
            try: verify('g2p.bin',v8)
            except AssertionError: pass
            else: raise AssertionError('V8 substitution escaped identity check')
    return manifest

def compare_runs(root,voices,graphs=True):
    vocab=json.loads((ROOT/'tests/fixtures/vocab.json').read_text())['vocab']
    for case in root.glob('*/*/stats.json'):
        d=case.parent; ps=(d/'phonemes.txt').read_text()
        assert np.array_equal(np.fromfile(d/'tokens.i32',dtype='<i4'),[0]+[vocab[c] for c in ps if c in vocab]+[0]),'tokens'
        voice='af_heart'
        pack=np.fromfile(voices/(voice+'.bin'),dtype='<f4').reshape(510,256)
        actual=np.fromfile(d/'style.f32',dtype='<f4')
        assert np.array_equal(actual,pack[len(ps)-1]),'official style row'
        assert not np.array_equal(actual,pack[len(ps)]),'shifted row control'
        st=json.loads(case.read_text());n=st['true_frames']
        assert st['decode_frames']==n and st['upsample_frames']==2*n and st['stft_samples']==600*n,'true lengths'
        audio=np.fromfile(d/'audio.f32',dtype='<f4');assert len(audio)==600*n and np.isfinite(audio).all()
    def difference(a,b):
        x=np.fromfile(a/'audio.f32',dtype='<f4');y=np.fromfile(b/'audio.f32',dtype='<f4')
        return dict(same_samples=len(x)==len(y),max_abs=float(np.max(np.abs(x-y))) if len(x)==len(y) else None,
                    rms=float(np.sqrt(np.mean((x.astype(float)-y)**2))) if len(x)==len(y) else None)
    first=root/'first';a=first/'a0'
    if graphs:
        assert json.loads((first/'a1/stats.json').read_text())['decode_hits']>json.loads((a/'stats.json').read_text())['decode_hits'],'replay exercised'
    report={name:difference(a,first/name) for name in ('a1','a2','a3','a4','b','style_change','aba','isolated','after_growth')}
    report['recreated']=difference(a,root/'recreated/a0')
    for name in ('key_b','key_style','key_aba'):
        report[name]=difference(first/'key_a',first/name)
    assert report['key_b']['rms']>0 and report['key_style']['rms']>0, 'stale decode inputs'
    for name in ('b','style_change'):
        assert not report[name]['same_samples'] or report[name]['rms']>0,'stale output control'
    for name in ('a1','a2','a3','a4','aba','isolated','after_growth'):
        assert np.array_equal(np.fromfile(a/'rounded.i32',dtype='<i4'),np.fromfile(first/name/'rounded.i32',dtype='<i4')),'duration state changed'
    # Report floating-point variation without inventing an acoustic pass threshold.
    report['claim']='Functional/cache/length checks only. Waveform variation is measured, not an equivalence gate.'
    return report

def invalid_artifacts(runtime,binary,args):
    # Exercise the same in-memory loaders using a development-only file adapter.
    paths_ = paths(args)
    good=[paths_['weights'],paths_['g2p'],paths_['voices']/'af_heart.bin']
    with tempfile.TemporaryDirectory() as directory:
        d=Path(directory)
        for i,label in enumerate(('weights','g2p','voice')):
            for suffix,payload in [('empty',b''),('truncated',good[i].read_bytes()[:128]),
                                   ('short_payload',good[i].read_bytes()[:-4096])]:
                bad=d/(label+suffix);bad.write_bytes(payload);before=sha256(bad)
                files=good.copy();files[i]=bad
                p=subprocess.run([str(runtime),'--check-assets',*map(str,files)],capture_output=True,timeout=60)
                assert p.returncode==1 and b'FAIL:' in p.stderr,(label,suffix,p.returncode,p.stderr)
                assert sha256(bad)==before,'explicit artifact overwritten'
        for flag in ('--weights','--g2p','--voices','--voice'):
            p=subprocess.run([str(binary),'Hello.',flag,'obsolete'],capture_output=True,timeout=30)
            assert p.returncode==1 and b'unknown option' in p.stderr,(flag,p.stderr)

def http_checks(binary,args,out):
    results={}
    with server(binary,args,out/'server.log') as (base,startup):
        status,_,page,_=request(base,'/');assert status==200
        assert b'<select' not in page and b"$('voice')" not in page, 'obsolete voice selector'
        for endpoint in ('/synthesize','/synthesize/stream'):
            for voice in ('af_heart','af_bella','af_nicole','af_sky'):
                status,_,body,_=request(base,endpoint,{'text':'Hello.','voice':voice})
                assert status==400 and 'voice' in json.loads(body)['error'], 'retired voice accepted'
        for endpoint in ('/synthesize','/synthesize/stream'):
            for payload in ({},{'text':''},{'text':' \t\n'},{'text':'Hello','voice':'absent'},{'text':'Hello','voice':'bad\x01'},{'text':'☃','input':'phonemes'}):
                status,_,body,_=request(base,endpoint,payload)
                assert status==400 and json.loads(body)['error'],(endpoint,payload,status)
            for raw in (b'{"text":"\xff"}',b'{"text":"x"',b'{"text":3}',b'{"text":"x","text":"y"}',b'{"text":"\\ud800"}'):
                status,_,_,_=request(base,endpoint,raw=raw);assert status==400,(raw,status)
        # JSON Unicode escapes must decode; they must not be read as literal uXXXX text.
        status,_,data,_=request(base,'/synthesize',raw=b'{"text":"h\\u025bl\\u02c8o\\u028a","input":"phonemes"}')
        assert status==200;wav_info(data)
        for voice in VOICES:
            status,headers,data,ms=request(base,'/synthesize',{'text':'Hello world.'});assert status==200
            results[voice]=dict(wav_info(data),headers=headers,wall_ms=ms);(out/(voice+'.wav')).write_bytes(data)
        # Buffered PCM16 is clamp[-1,1], multiply32767, truncate toward zero.
        status,_,data,_=request(base,'/synthesize/stream',{'text':'Hello world.'});assert status==200
        pcm=np.frombuffer(data,dtype='<f4');assert len(pcm)>0 and np.isfinite(pcm).all()
        buffered=np.frombuffer((out/'af_heart.wav').read_bytes(),dtype='<i2',offset=44).astype(float)/32767
        assert len(pcm)==len(buffered),'buffered/streaming sample count'
        results['stream_vs_buffered_rms']=float(np.sqrt(np.mean((pcm-buffered)**2)))
        results['stream_comparison']='Different inference calls; see repeatability measurements, not a sample-equivalence gate.'
        results['stream_samples']=len(pcm);(out/'stream.f32').write_bytes(data)
        # Disconnect early, then ensure the worker serves the next request.
        hostport=base.split('//')[1];host,port=hostport.split(':')
        conn=http.client.HTTPConnection(host,int(port),timeout=120)
        conn.request('POST','/synthesize/stream',body=json.dumps({'text':'Hello world. '*150}),headers={'Content-Type':'application/json'})
        resp=conn.getresponse();assert resp.status==200;resp.read(128);conn.close()
        status,_,data,_=request(base,'/synthesize',{'text':'After cancellation.'});assert status==200;wav_info(data)
        results['stats']=json.loads(request(base,'/stats')[2])
    return results

def main():
    ap=argparse.ArgumentParser(description=__doc__);model_args(ap)
    ap.add_argument('--official',type=Path,required=True);ap.add_argument('--output',type=Path,default=ROOT/'tests/results/gpu.json')
    ap.add_argument('--skip-semantic',action='store_true',help='reuse a separately saved semantic audit; still check artifact identities')
    args=ap.parse_args();idle_gpu();manifest=verify_identity(args)
    if not args.skip_semantic:
        subprocess.run([sys.executable,str(ROOT/'tests/artifacts.py'),'--official',str(args.official),'--models',str(args.models),
                        *(['--g2p',str(args.g2p)] if args.g2p else []),*(['--voices',str(args.voices)] if args.voices else [])],check=True)
    report=dict(provenance=provenance(),artifacts=manifest)
    binary=ROOT/'rokoko';runtime=ROOT/'tests/runtime';out=args.output.parent/'rokoko';out.mkdir(parents=True,exist_ok=True)
    info=embedded_identity(binary)
    assert info['files']==manifest['files'] and info['precision']=='fp16'
    assert len(info['files'])==3 and info['voice']=='af_heart'
    invalid_artifacts(runtime,binary,args)
    subprocess.run([str(binary),'Hello world.','-o',str(out/'cli.wav')],check=True,timeout=30)
    wav_info((out/'cli.wav').read_bytes())
    p=paths(args)
    subprocess.run([str(runtime),str(out/'library-af_heart')],check=True,timeout=120)
    row=dict(provenance=provenance(binary,bundled=True),repeatability=compare_runs(out/'library-af_heart',p['voices']),http=http_checks(binary,args,out))
    # Long normalized input, including raw abbreviations that expand past capacity.
    row['long']=[]
    for label,text in [('2047',('hello world '*171)[:2047]),('2048',('hello world '*171)[:2048]),('2049',('hello world '*171)[:2049]),('expand','Dr. Smith paid $123.45. '*120),
                       ('5000',('Hello world. '*400)[:5000]+' Final words purple lantern.'),
                       ('20000',('Hello world. '*1600)[:20000]+' Final words purple lantern.'),
                       ('no_sentence_delimiters',('hello world '*1667)[:20000]+' final words purple lantern')]:
        folder=out/label;folder.mkdir(exist_ok=True);source=folder/'input.txt';source.write_text(text)
        subprocess.run([str(runtime),str(folder),str(source)],check=True,timeout=240)
        normalized=(folder/'normalized.txt').read_bytes();prev=0
        for line in (folder/'spans.tsv').read_text().splitlines():
            a,b=map(int,line.split());assert prev<=a<b<=len(normalized) and not normalized[prev:a].decode().strip();assert len(normalized[a:b].decode())<=2048;prev=b
        assert not normalized[prev:].decode().strip()
        chunks=sorted(folder.glob('*/phonemes.txt'),key=lambda x:int(x.parent.name));assert chunks
        row['long'].append(dict(case=label,normalized_chars=len(normalized.decode()),chunks=len(chunks),last_phonemes=chunks[-1].read_text()))
    report['runtime']=row;save_report(args.output,report)
    print('PASS artifact identity, input/style/length, library lifecycle, CLI errors, HTTP and long-input coverage. Numerical/audio limitations remain in the report.')
if __name__=='__main__':main()
