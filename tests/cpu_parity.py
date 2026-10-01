"""Bounded CPU/CUDA inference comparison using the same bundled assets and fixed corpus.

These are engineering regression tolerances, not a perceptual equivalence claim.
Waveform samples can differ because reductions and sine phase are not bit-identical.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import wave
import numpy as np
from support import ROOT, embedded_identity, provenance, save_report, sha256
from bench import TEXTS


def audio_metrics(cpu, gpu):
    assert cpu.shape == gpu.shape and len(cpu), 'audio length'
    assert np.isfinite(cpu).all() and np.isfinite(gpu).all(), 'nonfinite audio'
    x=cpu.astype(np.float64); y=gpu.astype(np.float64)
    rms=lambda a: float(np.sqrt(np.mean(a*a)))
    window=np.hanning(1024)
    def spectrum(a):
        padded=np.pad(a,(512,512))
        frames=np.lib.stride_tricks.sliding_window_view(padded,1024)[::256]
        return np.abs(np.fft.rfft(frames*window,axis=1))
    sx,sy=spectrum(x),spectrum(y)
    return dict(samples=len(x),audio_seconds=len(x)/24000,rms_cpu=rms(x),rms_gpu=rms(y),
                rms_ratio=rms(x)/rms(y),waveform_rms_difference=rms(x-y),
                correlation=float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else 0.,
                spectral_relative_error=float(np.linalg.norm(sx-sy)/np.linalg.norm(sy)))


def check_audio(metrics):
    assert .9 <= metrics['rms_ratio'] <= 1.1, ('audio RMS ratio',metrics)
    assert metrics['spectral_relative_error'] <= .1, ('spectral relative error',metrics)


def write_wav(path,audio):
    with wave.open(str(path),'wb') as w:
        w.setparams((1,2,24000,0,'NONE','not compressed'))
        w.writeframes((np.clip(audio,-1,1)*32767).astype('<i2').tobytes())


def check_g2p(cpu, gpu, asset, out):
    lines=(ROOT/'tests/frontend/plain_test.txt').read_text().splitlines()[::10]
    source='\n'.join(lines)+'\n';outputs={}
    for name,binary in (('cpu',cpu),('cuda',gpu)):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='') if name=='cpu' else None
        result=subprocess.run([str(binary),str(asset),'--normalize'],input=source,
                              text=True,capture_output=True,check=True,timeout=180,env=env)
        outputs[name]=result.stdout.splitlines()
        (out/('g2p-'+name+'.txt')).write_text(result.stdout)
    assert len(outputs['cpu'])==len(outputs['cuda'])==len(lines)
    mismatches=[dict(text=t,cpu=a,cuda=b) for t,a,b in zip(lines,outputs['cpu'],outputs['cuda']) if a!=b]
    assert not mismatches, ('G2P pronunciation differences',mismatches)
    return dict(cases=len(lines),differences=mismatches)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cpu-runtime',type=Path,default=ROOT/'build/cpu/rokoko_runtime')
    ap.add_argument('--gpu-runtime',type=Path,default=ROOT/'tests/runtime')
    ap.add_argument('--cpu',type=Path,default=ROOT/'build/cpu/rokoko')
    ap.add_argument('--gpu',type=Path,default=ROOT/'rokoko')
    ap.add_argument('--cpu-g2p',type=Path,default=ROOT/'build/cpu/rokoko_g2p_check')
    ap.add_argument('--gpu-g2p',type=Path,default=ROOT/'tests/frontend/g2p_check')
    ap.add_argument('--g2p-asset',type=Path,default=ROOT/'build/assets/g2p.bin')
    ap.add_argument('--output',type=Path,default=ROOT/'tests/results/cpu-parity')
    args=ap.parse_args()
    for key in ('cpu_runtime','gpu_runtime','cpu','gpu','output','cpu_g2p','gpu_g2p','g2p_asset'): setattr(args,key,getattr(args,key).resolve())
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    cpu_info=embedded_identity(args.cpu);gpu_info=embedded_identity(args.gpu)
    assert cpu_info['backend']=='cpu' and gpu_info['backend']=='cuda'
    assert cpu_info['files']==gpu_info['files'], 'different model assets'
    # The CPU executable must run without loading either CUDA or a dynamic BLAS.
    dependencies=subprocess.check_output(['ldd',str(args.cpu)],text=True)
    assert not any(s in dependencies.lower() for s in ('cuda','cublas','openblas'))
    cases=json.loads((ROOT/'tests/fixtures/quality.json').read_text())['cases']
    cases += [dict(id=f'benchmark_{i}',text=text) for i,text in enumerate(TEXTS)]
    report=dict(cpu=provenance(args.cpu,bundled=True),gpu=provenance(args.gpu,bundled=True),
                cpu_threads=os.environ.get('ROKOKO_CPU_THREADS','8 (default maximum)'),
                tolerances=dict(duration_max_abs=.02,f0_rmse_hz=.25,noise_rmse=.01,
                                audio_rms_ratio=[.9,1.1],spectral_relative_error=.1),cases=[])
    assert sha256(args.g2p_asset)==cpu_info['files']['g2p.bin']['sha256']
    report['g2p']=check_g2p(args.cpu_g2p,args.gpu_g2p,args.g2p_asset,out)
    print('PASS CPU/CUDA G2P on',report['g2p']['cases'],'corpus sentences',flush=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    for case in cases:
        d=out/case['id'];d.mkdir(parents=True,exist_ok=True);textfile=d/'text.txt';textfile.write_text(case['text'])
        for kind,binary in [('cpu',args.cpu_runtime),('gpu',args.gpu_runtime)]:
            with (d/(kind+'.log')).open('w') as log:
                subprocess.run([str(binary),str(d/kind),str(textfile)],check=True,stdout=log,stderr=log,
                               timeout=300,env=env if kind=='cpu' else None)
        assert (d/'cpu/normalized.txt').read_bytes()==(d/'gpu/normalized.txt').read_bytes()
        chunks=sorted((d/'cpu').glob('*/stats.json'));other=sorted((d/'gpu').glob('*/stats.json'))
        assert len(chunks)==len(other)>0,'chunk count'
        row=dict(id=case['id'],text=case['text'],chunks=[])
        for p,q in zip(chunks,other):
            a,b=p.parent,q.parent
            for name in ('phonemes.txt','tokens.i32','style.f32','rounded.i32'):
                assert (a/name).read_bytes()==(b/name).read_bytes(),(case['id'],name)
            metrics={}
            for name in ('duration','f0','noise'):
                x=np.fromfile(a/(name+'.f32'),dtype='<f4').astype(float);y=np.fromfile(b/(name+'.f32'),dtype='<f4').astype(float)
                assert x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all()
                metrics[name]=dict(max_abs=float(np.max(np.abs(x-y))),rmse=float(np.sqrt(np.mean((x-y)**2))))
            assert metrics['duration']['max_abs']<=.02,(case['id'],'duration',metrics)
            assert metrics['f0']['rmse']<=.25,(case['id'],'f0',metrics)
            assert metrics['noise']['rmse']<=.01,(case['id'],'noise',metrics)
            x=np.fromfile(a/'audio.f32',dtype='<f4');y=np.fromfile(b/'audio.f32',dtype='<f4')
            metrics['audio']=audio_metrics(x,y);check_audio(metrics['audio'])
            for path,audio in [(a,x),(b,y)]:
                stats=json.loads((path/'stats.json').read_text())
                assert len(audio)==stats['true_frames']*600==stats['stft_samples']
                write_wav(path/'audio.wav',audio)
            # Negative controls prove silence/gain errors cannot pass this comparison.
            for bad in (np.zeros_like(x),y*.5):
                try: check_audio(audio_metrics(bad,y))
                except AssertionError: pass
                else: raise AssertionError('corrupt audio control escaped')
            row['chunks'].append(metrics)
        report['cases'].append(row);save_report(out/'report.json',report)
        print('PASS',case['id'], 'spectral error', max(m['audio']['spectral_relative_error'] for m in row['chunks']),flush=True)
    links=[]
    import html
    for case in cases:
        links.append(f'<h2>{html.escape(case["text"])}</h2>')
        for kind in ('cpu','gpu'):
            for p in sorted((out/case['id']/kind).glob('*/audio.wav')):
                links.append(f'<p>{kind.upper()} <audio controls src="{p.relative_to(out)}"></audio></p>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>CPU and GPU listening comparison</title><h1>Rokoko CPU / GPU</h1>'+''.join(links))
    print(f'PASS {len(cases)} CPU/CUDA cases; audio and report in {out}')

if __name__=='__main__': main()
