"""Untouched official KModel comparison: capture only, with hook neutrality checks."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from kokoro.model import KModel
from support import ROOT, provenance, save_report, sha256, idle_gpu

def load_model(official):
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=False
    return KModel(repo_id='hexgrad/Kokoro-82M',config=str(official/'config.json'),model=str(official/'kokoro-v1_0.pth')).eval().cuda()

def run(model,ps,style, capture=False, reinject=None):
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    saved={};handles=[];original=model.predictor.F0Ntrain
    if capture:
        def duration(_module,_args,out): saved['duration']=torch.sigmoid(out).sum(-1).detach().cpu().numpy().ravel()
        handles.append(model.predictor.duration_proj.register_forward_hook(duration))
        def prosody(en,s):
            saved['en']=en.detach().clone();saved['s']=s.detach().clone()
            if reinject is not None:
                assert torch.equal(en,reinject['en']) and torch.equal(s,reinject['s']), 'reference capture input drift'
                en,s=reinject['en'],reinject['s']
            f0,n=original(en,s)
            saved['f0']=f0.detach().cpu().numpy().ravel();saved['noise']=n.detach().cpu().numpy().ravel()
            return f0,n
        model.predictor.F0Ntrain=prosody
    try: result=model(ps,style,return_output=True)
    finally:
        model.predictor.F0Ntrain=original
        for h in handles:h.remove()
    saved.update(audio=result.audio.numpy(),rounded=result.pred_dur.numpy())
    return saved

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--official',type=Path,required=True)
    ap.add_argument('--results',type=Path,default=ROOT/'tests/results');ap.add_argument('--output',type=Path,default=ROOT/'tests/results/reference.json')
    args=ap.parse_args();idle_gpu();model=load_model(args.official)
    report=dict(provenance=provenance(),torch=torch.__version__,cudnn_tf32=True,matmul_tf32=False,seed=42,
                claim='Exact input/length assertions; numerical differences reported without an uncalibrated parity threshold.',cases=[])
    folders=[args.results/'rokoko/library-af_heart/first'/x for x in ('a0','b','style_af_heart')]
    for folder in folders:
        ps=(folder/'phonemes.txt').read_text();voice=folder.name[6:] if folder.name.startswith('style_') else 'af_heart'
        pack=torch.load(args.official/'voices'/(voice+'.pt'),map_location='cpu',weights_only=True)
        style=pack[len(ps)-1].cuda()
        assert np.array_equal(style.cpu().numpy().ravel(),np.fromfile(folder/'style.f32',dtype='<f4'))
        plain=run(model,ps,style);captured=run(model,ps,style,True);reinjected=run(model,ps,style,True,captured)
        assert np.array_equal(plain['audio'],captured['audio']), 'instrumentation changes official output'
        assert np.array_equal(plain['audio'],reinjected['audio']), 're-injection changes official output'
        row=dict(binary='rokoko',case=folder.name,voice=voice,phonemes=ps,instrumentation='neutral; full prosody inputs re-injected',errors={})
        for field,dtype in [('duration','<f4'),('rounded','<i4'),('f0','<f4'),('noise','<f4')]:
            actual=np.fromfile(folder/(field+('.i32' if dtype=='<i4' else '.f32')),dtype=dtype);expected=captured[field]
            row['errors'][field]=dict(rokoko_count=len(actual),official_count=len(expected))
            if len(actual)==len(expected):
                delta=actual.astype(float)-expected.astype(float)
                row['errors'][field].update(max_abs=float(np.max(np.abs(delta))),rms=float(np.sqrt(np.mean(delta**2))))
                if field in ('duration','rounded'): row['errors'][field].update(rokoko=actual.tolist(),official=expected.tolist(),delta=delta.tolist())
            expected.astype(dtype).tofile(folder/('official_'+field+('.i32' if dtype=='<i4' else '.f32')))
        report['cases'].append(row)
    save_report(args.output,report);print('PASS reference instrumentation and exact styles. Numerical differences recorded, not waived.')
if __name__=='__main__':main()
