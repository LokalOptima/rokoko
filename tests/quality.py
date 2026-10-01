"""Fixed paired intelligibility report. ASR/scoring happen outside synthesis timing."""
import argparse
import gc
import json
import subprocess
import time
import wave
from pathlib import Path
import numpy as np
import torch
from kokoro import KPipeline
from reference import load_model
from scoring import score, paired_interval
from support import ROOT, idle_gpu, model_args, server, request, wav_info, save_report, sha256, provenance
VOICES=('af_heart',)
def wav(path,audio):
    pcm=(np.clip(audio,-1,1)*32767).astype('<i2')
    with wave.open(str(path),'wb') as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(24000);f.writeframes(pcm.tobytes())

def main():
    ap=argparse.ArgumentParser(description=__doc__);model_args(ap)
    ap.add_argument('--official',type=Path,required=True);ap.add_argument('--asr',type=Path,required=True)
    ap.add_argument('--asr-weights',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=ROOT/'tests/results/quality.json')
    args=ap.parse_args();idle_gpu()
    fixture=ROOT/'tests/fixtures/quality.json';cases=json.loads(fixture.read_text())['cases'];out=args.output.parent/'quality-audio';out.mkdir(parents=True,exist_ok=True)
    report=dict(provenance=provenance(),asr=provenance(args.asr),asr_weights_sha256=sha256(args.asr_weights),
                fixture_sha256=sha256(fixture),cases=cases,rows=[],summary={},
                limitations=['Small fixed diagnostic set; no perceptual or WER equivalence claim.',
                'Literal scoring preserves digits versus spelled-number differences in ASR; inspect transcripts.',
                'Reviewed frontend cases can overlap G2P training; not a held-out generalization estimate.',
                'No acceptance margin selected; paired intervals are descriptive.'])
    normalized=subprocess.run([str(ROOT/'tests/frontend/normalize_cli')],input='\n'.join(c['text'] for c in cases)+'\n',text=True,capture_output=True,check=True).stdout
    ps_lines=subprocess.run([str(ROOT/'tests/frontend/g2p_check'),str(args.g2p or args.models/'g2p.bin')],input=normalized,text=True,capture_output=True,check=True).stdout.splitlines()
    assert len(ps_lines)==len(cases)
    for c,ps in zip(cases,ps_lines):
        raw=subprocess.run([str(ROOT/'tests/helpers'),'chunks'],input=ps,text=True,capture_output=True,check=True).stdout
        c['rokoko_chunks']=[bytes.fromhex(line.split()[2]).decode() for line in raw.splitlines()];assert c['rokoko_chunks']
    report['rokoko']=provenance(ROOT/'rokoko',bundled=True)
    system='rokoko'
    with server(ROOT/system,args,out/(system+'.log')) as (base,startup):
        for voice in VOICES:
            for c in cases:
                status,h,data,ms=request(base,'/synthesize',dict(text=c['text']));assert status==200,(system,c,status,data)
                path=out/(system+'-'+voice+'-'+c['id']+'.wav');path.write_bytes(data)
                report['rows'].append(dict(system=system,voice=voice,id=c['id'],wav=str(path),wall_ms=ms,telemetry=h,**wav_info(data)))
    model=load_model(args.official);pipeline=KPipeline(lang_code='a',model=model,repo_id='hexgrad/Kokoro-82M',trf=False)
    for voice in VOICES:
        pack=torch.load(args.official/'voices'/(voice+'.pt'),weights_only=True)
        for c in cases:
            for system in ('kokoro','kokoro_same_phonemes'):
                torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.cuda.synchronize();t=time.perf_counter()
                if system=='kokoro':
                    result=list(pipeline(c['text'],voice=pack));audio=np.concatenate([x.audio.numpy() for x in result]);chunks=[x.phonemes for x in result]
                else:
                    chunks=c['rokoko_chunks'];audio=np.concatenate([model(ps,pack[len(ps)-1]).numpy() for ps in chunks])
                torch.cuda.synchronize();elapsed=(time.perf_counter()-t)*1000
                path=out/(system+'-'+voice+'-'+c['id']+'.wav');wav(path,audio)
                report['rows'].append(dict(system=system,voice=voice,id=c['id'],chunks=chunks,wav=str(path),wall_ms=elapsed,**wav_info(path.read_bytes())))
    del pipeline,model;gc.collect();torch.cuda.empty_cache()
    # One ASR process, fixed file order. No text corrector enabled.
    cmd=[str(args.asr.resolve()),'--weights',str(args.asr_weights.resolve()),*[r['wav'] for r in report['rows']]]
    process=subprocess.run(cmd,text=True,capture_output=True,check=True,timeout=1200)
    (out/'asr.log').write_text(process.stderr);transcripts=process.stdout.splitlines()
    assert len(transcripts)==len(report['rows']),(len(transcripts),len(report['rows']))
    by_id={c['id']:c for c in cases}
    for row,hyp in zip(report['rows'],transcripts):row.update(transcript=hyp,**score(by_id[row['id']]['spoken'],hyp))
    for voice in VOICES:
        group={s:[r for r in report['rows'] if r['voice']==voice and r['system']==s] for s in ('rokoko','kokoro','kokoro_same_phonemes')}
        for system,rows in group.items():
            report['summary'][voice+'/'+system]=dict(errors=sum(r['errors'] for r in rows),words=sum(r['words'] for r in rows),
                wer=sum(r['errors'] for r in rows)/sum(r['words'] for r in rows),sentence_errors=sum(r['errors']>0 for r in rows),sentences=len(rows),
                delta_vs_kokoro_95=paired_interval(rows,group['kokoro']),delta_vs_same_phonemes_95=paired_interval(rows,group['kokoro_same_phonemes']))
    # Explicit controls: negation, wrong numbers, and omitted tail must remain errors.
    for ref,hyp in [('do not go','do go'),('five birds','six birds'),('the final words purple lantern','the final words')]:assert score(ref,hyp)['errors']>0
    save_report(args.output,report);print(json.dumps(report['summary'],indent=2));print('Saved',args.output)
if __name__=='__main__':main()
