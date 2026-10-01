"""Compare experimental INT8 output with hash-verified saved CPU/GPU traces."""
import argparse
import html
import json
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpu_parity import audio_metrics, check_audio, write_wav
from support import ROOT, save_report, sha256, provenance
from scoring import score

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference',type=Path,default=ROOT/'tests/results/cpu-rtfx-parity')
    ap.add_argument('--baseline',type=Path,default=ROOT/'build/cpu/rokoko')
    ap.add_argument('--experiment',type=Path,default=ROOT/'build/int8')
    ap.add_argument('--output',type=Path,default=ROOT/'tests/results/int8-quality')
    ap.add_argument('--asr',type=Path)
    ap.add_argument('--asr-weights',type=Path)
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    ref=args.reference.resolve();exp=args.experiment.resolve()
    saved=json.loads((ref/'report.json').read_text())
    assert sha256(args.baseline)==saved['cpu']['binary_sha256'],'baseline traces belong to a different binary'
    report=dict(experiment=json.loads((exp/'build.json').read_text()),
                provenance=provenance(exp/'rokoko.int8',bundled=True),
                reference_report_sha256=sha256(ref/'report.json'),
                baseline_sha256=sha256(args.baseline),cases=[])
    pages=['<!doctype html><meta charset="utf-8"><title>INT8 comparison</title>',
           '<style>body{font:17px sans-serif;max-width:900px;margin:40px auto}audio{width:100%}</style>',
           '<h1>FP32 CPU and experimental INT8</h1><p>Same text, voice and playback gain. INT8 applies only to vocoder residual convolutions.</p>']
    for case in saved['cases']:
        d=out/case['id'];d.mkdir(exist_ok=True);text=d/'text.txt';text.write_text(case['text'])
        with (d/'runtime.log').open('w') as log:
            subprocess.run([str(exp/'runtime'),str(d/'int8'),str(text)],stdout=log,stderr=log,check=True,timeout=300)
        assert (d/'int8/normalized.txt').read_bytes()==(ref/case['id']/'cpu/normalized.txt').read_bytes()
        row=dict(id=case['id'],text=case['text'],chunks=[])
        pages.append('<h2>'+html.escape(case['text'])+'</h2>')
        for chunk in sorted((d/'int8').glob('*/audio.f32')):
            c=chunk.parent;cpu=ref/case['id']/'cpu'/c.name;gpu=ref/case['id']/'gpu'/c.name
            for name in ('phonemes.txt','tokens.i32','style.f32','duration.f32','rounded.i32','f0.f32','noise.f32'):
                assert (c/name).read_bytes()==(cpu/name).read_bytes(),(case['id'],name,'changed outside vocoder')
            x=np.fromfile(chunk,dtype='<f4');write_wav(c/'audio.wav',x)
            metrics={}
            for kind,path in [('fp32',cpu),('gpu',gpu)]:
                y=np.fromfile(path/'audio.f32',dtype='<f4');m=audio_metrics(x,y)
                try: check_audio(m);m['within_existing_audio_limits']=True
                except AssertionError: m['within_existing_audio_limits']=False
                metrics[kind]=m
            row['chunks'].append(metrics)
            baseline=d/('fp32-'+c.name+'.wav');shutil.copyfile(cpu/'audio.wav',baseline)
            for label,path in [('FP32 CPU',baseline),('INT8',c/'audio.wav')]:
                pages.append(f'<p>{label}<audio controls preload="none" src="{path.relative_to(out)}"></audio></p>')
        report['cases'].append(row);save_report(out/'report.json',report)
        print(case['id'],[(k,round(max(c[k]['spectral_relative_error'] for c in row['chunks']),5),
              all(c[k]['within_existing_audio_limits'] for c in row['chunks'])) for k in ('fp32','gpu')],flush=True)
    (out/'index.html').write_text(''.join(pages))
    if args.asr:
        assert args.asr_weights
        fixtures=json.loads((ROOT/'tests/fixtures/quality.json').read_text())['cases']
        fixtures += [dict(id=c['id'], spoken=c['text']) for c in saved['cases'] if c['id'].startswith('benchmark_')]
        recordings=[]
        for case in fixtures:
            for kind in ('fp32','int8'):
                d=out/case['id'];files=sorted(d.glob('fp32-*.wav') if kind=='fp32' else d.glob('int8/*/audio.wav'))
                assert len(files)==1,'ASR fixture must have one chunk'
                recordings.append(dict(id=case['id'],system=kind,reference=case['spoken'],wav=str(files[0])))
        result=subprocess.run([str(args.asr.resolve()),'--weights',str(args.asr_weights.resolve()),
                               *[x['wav'] for x in recordings]],capture_output=True,text=True,check=True,timeout=600)
        (out/'asr.log').write_text(result.stderr);transcripts=result.stdout.splitlines()
        assert len(transcripts)==len(recordings),(len(transcripts),len(recordings))
        for row,hyp in zip(recordings,transcripts):row.update(transcript=hyp,**score(row['reference'],hyp))
        report['asr']=dict(binary_sha256=sha256(args.asr),weights_sha256=sha256(args.asr_weights),rows=recordings,
            summary={kind:dict(errors=sum(x['errors'] for x in recordings if x['system']==kind),
                                words=sum(x['words'] for x in recordings if x['system']==kind)) for kind in ('fp32','int8')})
        print('ASR',report['asr']['summary'],flush=True);save_report(out/'report.json',report)
    print('Listening page:',out/'index.html',flush=True)

if __name__=='__main__':main()
