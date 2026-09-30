"""Independently derive stored tensors from official Kokoro. Never imports the converter."""
import argparse
import hashlib
import json
import math
import re
import struct
from pathlib import Path
import numpy as np
import torch
from kokoro.model import KModel
from support import ROOT, model_args, provenance, save_report, sha256

VOICES = ('af_heart',)

def read_koko(path):
    data=path.read_bytes()
    if len(data)<16: raise ValueError('truncated weight header')
    magic,version,length=struct.unpack_from('<IIQ',data)
    if magic != 0x4f4b4f4b or version not in (1,2) or length > len(data)-16:
        raise ValueError('invalid weight header')
    start=(16+length+4095)//4096*4096
    tensors={}; spans=[]
    for line in data[16:16+length].decode('ascii').splitlines():
        name,off,size,dtype,*dims=line.split(); off=int(off);size=int(size);shape=tuple(map(int,dims));dtype={'float16':'fp16','float32':'fp32'}.get(dtype,dtype)
        if name in tensors or dtype not in ('fp16','fp32') or not shape or min(shape)<1: raise ValueError('invalid tensor '+name)
        width=2 if dtype=='fp16' else 4
        if size != math.prod(shape)*width or off<0 or off%width or start+off+size>len(data): raise ValueError('tensor bounds '+name)
        tensors[name]=np.frombuffer(data,dtype='<f2' if width==2 else '<f4',count=size//width,offset=start+off).reshape(shape)
        spans.append((off,off+size))
    spans.sort()
    if any(a[1]>b[0] for a,b in zip(spans,spans[1:])): raise ValueError('overlapping tensors')
    return version,tensors

def derive(model):
    """Infer the storage policy from official module types, independent of export code.

    Computed weight norm uses a float64 center and a conservative forward-error interval:
    gamma_(n+3) for an n-term float32 reduction, plus four unit roundoffs for sqrt/div/mul.
    Quantization is applied to interval endpoints, rather than a blanket FP16 ULP allowance.
    """
    state=model.state_dict(); modules=dict(model.named_modules()); out={}
    u=np.finfo(np.float32).eps/2
    for name,tensor in state.items():
        value=tensor.detach().cpu().numpy()
        parent,field=name.rsplit('.',1); module=modules[parent]
        if parent=='bert.pooler': continue  # unused by KModel.forward_with_tokens
        if field=='weight_g' or field.startswith('bias_hh_l'): continue
        if isinstance(module,torch.nn.LSTM) and field.startswith('bias_ih_l'):
            reverse='_reverse' if field.endswith('_reverse') else ''
            other=state[parent+'.bias_hh_l0'+reverse].cpu().numpy()
            arr=(value+other).astype(np.float32)
            out[parent+'.bias_combined_'+('rev' if reverse else 'fwd')]=(arr,arr,arr,'combined_bias')
            continue
        lo=hi=value; rule='exact'
        if field=='weight_v':
            v=value.astype(np.float64)
            g=state[parent+'.weight_g'].cpu().numpy().astype(np.float64)
            n=math.prod(v.shape[1:]); norms=np.sqrt(np.sum(v.reshape(v.shape[0],-1)**2,axis=1)).reshape(g.shape)
            if np.any(norms==0): raise ValueError('zero official norm '+name)
            value=v*(g/norms)
            gamma=(n+3)*u/(1-(n+3)*u)
            delta=np.abs(value)*(gamma+4*u)
            lo=value-delta;hi=value+delta;rule='float32_weight_norm_interval'
        key=name; dtype=np.float32
        if isinstance(module,torch.nn.LSTM) and field.startswith('weight_'):
            key += '.f16';dtype=np.float16
        elif isinstance(module,torch.nn.Linear) and field=='weight' and parent!='decoder.generator.m_source.l_linear':
            key += '.f16';dtype=np.float16
        elif isinstance(module,torch.nn.ConvTranspose1d) and field=='weight_v' and module.groups==1:
            key += '.f16';dtype=np.float16
        elif isinstance(module,torch.nn.Conv1d) and field in ('weight','weight_v') and parent not in (
                'decoder.F0_conv','decoder.N_conv','decoder.asr_res.0','predictor.F0_proj','predictor.N_proj'):
            dtype=np.float16
            if value.shape[0]%4:
                key += '.f16'
            else:
                value=value.transpose(0,2,1);lo=lo.transpose(0,2,1);hi=hi.transpose(0,2,1)
                pad=(-value.shape[-1])%8
                key += '.nhwc_f16'+('_pad'+str(value.shape[-1]+pad) if pad else '')
                if pad:
                    value=np.pad(value,((0,0),(0,0),(0,pad)))
                    lo=np.pad(lo,((0,0),(0,0),(0,pad)));hi=np.pad(hi,((0,0),(0,0),(0,pad)))
        # Outward endpoint rounding includes float32 computation/cast error where applicable.
        out[key]=(value.astype(dtype),lo.astype(dtype),hi.astype(dtype),rule)
    return out

def check_tensors(actual, expected):
    if set(actual)!=set(expected):
        raise AssertionError(f'tensor names: missing={sorted(set(expected)-set(actual))}, extra={sorted(set(actual)-set(expected))}')
    errors={}
    for name,(center,lo,hi,rule) in expected.items():
        value=actual[name]
        assert value.shape==center.shape and value.dtype==center.dtype, 'shape/dtype '+name
        assert np.isfinite(value).all(), 'nonfinite '+name
        assert np.logical_and(value>=lo,value<=hi).all(), 'value bound '+name
        errors[name]=dict(rule=rule, max_abs=float(np.max(np.abs(value.astype(np.float64)-center.astype(np.float64)))))
    return errors

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    model_args(ap)
    ap.add_argument('--official',type=Path,required=True)
    ap.add_argument('--manifest',type=Path,default=ROOT/'tests/fixtures/artifacts.json')
    ap.add_argument('--approve',action='store_true',help='write initial manifest only after independent semantic checks pass')
    ap.add_argument('--output',type=Path,default=ROOT/'tests/results/artifacts.json')
    ap.add_argument('--schema',type=Path,help='write independently derived C++ shape schema')
    args=ap.parse_args();torch.set_num_threads(2)
    model=KModel(repo_id='hexgrad/Kokoro-82M',config=str(args.official/'config.json'),model=str(args.official/'kokoro-v1_0.pth')).eval()
    fp32={k:(v.cpu().numpy(),v.cpu().numpy(),v.cpu().numpy(),'exact') for k,v in model.state_dict().items()}
    fp16=derive(model)
    report=dict(provenance=provenance(), torch=torch.__version__, files={}, checks={}, controls={})
    schema=[]
    for version,name,expected in [(1,'weights.bin',fp32),(2,'weights.fp16.bin',fp16)]:
        path=args.models/name;ver,actual=read_koko(path); assert ver==version
        report['checks'][name]=check_tensors(actual,expected)
        report['files'][name]=dict(size=path.stat().st_size,sha256=sha256(path))
        schema.append((version,expected))
        # The semantic oracle must reject an independently chosen damaged copied tensor.
        key='bert.embeddings.LayerNorm.bias'; broken=dict(actual);broken[key]=actual[key].copy();broken[key].flat[0]+=1
        try: check_tensors(broken,expected)
        except AssertionError as e:
            assert str(e).startswith('value bound'),str(e);report['controls'][name+'_corruption']='detected'
        else: raise AssertionError('semantic corruption was not detected')
    for voice in VOICES:
        path=(args.voices or args.models/'voices')/(voice+'.bin')
        official=torch.load(args.official/'voices'/(voice+'.pt'),map_location='cpu',weights_only=True).numpy()
        actual=np.fromfile(path,dtype='<f4')
        assert np.array_equal(actual,official.reshape(-1)),voice
        report['files']['voices/'+voice+'.bin']=dict(size=path.stat().st_size,sha256=sha256(path))
    cfg=json.loads((args.official/'config.json').read_text())
    assert cfg['vocab']==json.loads((ROOT/'tests/fixtures/vocab.json').read_text())['vocab']
    g2p=args.g2p or args.models/'g2p.bin'
    # Identity approved in frontend cleanup, independently matched to V11 checkpoint (see frontend README).
    assert hashlib.md5(g2p.read_bytes()).hexdigest()=='98dbb7bb697565d131a565ac644ae5da', 'expected verified G2P V11'
    report['files']['g2p.bin']=dict(size=g2p.stat().st_size,sha256=sha256(g2p))
    official_files=['config.json','kokoro-v1_0.pth']+['voices/'+v+'.pt' for v in VOICES]
    report['official']={f:dict(size=(args.official/f).stat().st_size,sha256=sha256(args.official/f)) for f in official_files}
    if args.approve:
        if args.manifest.exists(): raise RuntimeError('refusing to overwrite an approved manifest')
        save_report(args.manifest,dict(source='hexgrad/Kokoro-82M',source_revision=json.loads((ROOT/'tests/fixtures/vocab.json').read_text())['source_revision'],files=report['files'],official=report['official']))
    else:
        manifest=json.loads(args.manifest.read_text())
        assert manifest['files']==report['files'], 'artifact identity mismatch'
        assert manifest['official']==report['official'], 'official identity mismatch'
    if args.schema:
        lines=['// Derived from official KModel module shapes by tests/artifacts.py.','#pragma once','#include <initializer_list>',
               'namespace rokoko {','struct TensorSpec { const char* name; const char* dtype; std::initializer_list<int> shape; };']
        for version,expected in schema:
            lines.append(f'inline const TensorSpec MODEL_V{version}[] = {{')
            for name,(arr,*_) in sorted(expected.items()):
                lines.append('    {"'+name+'", "'+('fp16' if arr.dtype==np.float16 else 'fp32')+'", {'+', '.join(map(str,arr.shape))+'}},')
            lines.append('};')
        lines.append('} // namespace rokoko');args.schema.write_text('\n'.join(lines)+'\n')
    save_report(args.output,report)
    print(f'PASS: {len(fp32)} FP32 tensors, {len(fp16)} converted tensors, af_heart, vocabulary, G2P V11; corruption controls detected.')

if __name__=='__main__': main()
