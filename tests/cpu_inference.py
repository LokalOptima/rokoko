"""CPU-only library, corrupted-input, CLI/HTTP and deterministic state regression checks."""
import argparse
import os
from pathlib import Path
import subprocess
import numpy as np
from support import ROOT, model_args, paths, provenance, save_report
from gpu import compare_runs, invalid_artifacts, http_checks


def main():
    ap=argparse.ArgumentParser(description=__doc__);model_args(ap)
    ap.add_argument('--binary',type=Path,default=ROOT/'build/cpu/rokoko')
    ap.add_argument('--runtime',type=Path,default=ROOT/'build/cpu/rokoko_runtime')
    ap.add_argument('--output',type=Path,default=ROOT/'tests/results/cpu-inference')
    args=ap.parse_args();args.binary=args.binary.resolve();args.runtime=args.runtime.resolve();args.output.mkdir(parents=True,exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    subprocess.run([str(args.runtime),str(args.output/'library')],check=True,timeout=600)
    reports=compare_runs(args.output/'library',paths(args)['voices'],graphs=False)
    # The CPU implementation's reduction order is fixed. Exercise A -> B -> A,
    # retained arenas, and context recreation rather than only repeated identical calls.
    original=np.fromfile(args.output/'library/first/a0/audio.f32',dtype='<f4')
    for case in ['first/'+x for x in ('a1','a2','a3','a4','aba','isolated','after_growth')]+['recreated/a0']:
        actual=np.fromfile(args.output/'library'/case/'audio.f32',dtype='<f4')
        assert np.array_equal(original,actual),('CPU state drift',case)
    invalid_artifacts(args.runtime,args.binary,args)
    for value in ('0','65','invalid'):
        result=subprocess.run([str(args.binary),'Hello.'],env=dict(os.environ,ROKOKO_CPU_THREADS=value),capture_output=True,timeout=30)
        assert result.returncode==1 and b'ROKOKO_CPU_THREADS' in result.stderr
    http=http_checks(args.binary,args,args.output)
    save_report(args.output/'report.json',dict(provenance=provenance(args.binary,bundled=True),library=reports,http=http))
    print('PASS CPU library contracts, deterministic state, corrupt assets, CLI and HTTP/streaming recovery')

if __name__=='__main__':main()
