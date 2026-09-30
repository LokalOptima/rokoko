"""Official ordinary-user and identical-phoneme timings on the same fixed workload."""
import time
START=time.perf_counter()
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
from kokoro import KPipeline
from reference import load_model
from bench import TEXTS, summary
from quality import wav
from support import ROOT, idle_gpu, model_args, provenance, save_report

def main():
    ap=argparse.ArgumentParser(description=__doc__);model_args(ap);ap.add_argument('--official',type=Path,required=True);ap.add_argument('--repeats',type=int,default=5)
    args=ap.parse_args();idle_gpu();torch.set_num_threads(2)
    normalized=subprocess.run([str(ROOT/'tests/frontend/normalize_cli')],input='\n'.join(TEXTS)+'\n',text=True,capture_output=True,check=True).stdout
    phonemes=subprocess.run([str(ROOT/'tests/frontend/g2p_check'),str(args.g2p or args.models/'g2p.bin')],input=normalized,text=True,capture_output=True,check=True).stdout.splitlines()
    start=time.perf_counter();model=load_model(args.official);pipeline=KPipeline(lang_code='a',model=model,repo_id='hexgrad/Kokoro-82M',trf=False)
    pack=torch.load(args.official/'voices/af_heart.pt',weights_only=True);init_ms=(time.perf_counter()-start)*1000
    report=dict(provenance=provenance(),settings=dict(torch=torch.__version__,cudnn_tf32=True,matmul_tf32=False,compile=False,voice='af_heart',seed=42),
        model_pipeline_initialization_ms=init_ms,claim='Descriptive timings, no performance gate or audio equivalence claim. ASR is outside timing.',rows=[])
    out=ROOT/'tests/results/reference-bench';out.mkdir(parents=True,exist_ok=True)
    list(pipeline('Warmup.',voice=pack));torch.cuda.synchronize()
    for mode in ('pipeline','same_phonemes_model_only'):
        for i,(text,ps) in enumerate(zip(TEXTS,phonemes)):
            parts=subprocess.run([str(ROOT/'tests/helpers'),'chunks'],input=ps,text=True,capture_output=True,check=True).stdout
            chunks=[bytes.fromhex(line.split()[2]).decode() for line in parts.splitlines()]
            for repeat in range(args.repeats+1):
                torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.cuda.synchronize();start=time.perf_counter()
                if mode=='pipeline':
                    result=list(pipeline(text,voice=pack));audio=np.concatenate([x.audio.numpy() for x in result])
                else:audio=np.concatenate([model(p,pack[len(p)-1]).numpy() for p in chunks])
                torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
                report['rows'].append(dict(mode=mode,text_id=i,text=text,phonemes=ps,repeat=repeat,first=repeat==0,wall_ms=elapsed,samples=len(audio)))
                if repeat==0:wav(out/f'{mode}-{i}.wav',audio)
    report['summary']={f'{mode}/{i}':summary([r for r in report['rows'] if r['mode']==mode and r['text_id']==i and not r['first']]) for mode in ('pipeline','same_phonemes_model_only') for i in range(3)}
    report['gpu_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    save_report(ROOT/'tests/results/reference-bench.json',report);print(json.dumps(report['summary'],indent=2))
if __name__=='__main__':main()
