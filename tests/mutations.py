"""Isolated production-source padding mutation; failures must reach the length assertion."""
import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from support import ROOT, model_args, paths, save_report, provenance

def main():
    ap=argparse.ArgumentParser(description=__doc__);model_args(ap);args=ap.parse_args();report=dict(provenance=provenance(),controls={})
    cuda=Path(os.environ.get('CUDA_HOME','/usr/local/cuda-13.1'))
    objs=[ROOT/'src'/(x+'.o') for x in ('kernels','cutlass_conv','cutlass_gemm','cutlass_gemm_f16','cutlass_conv_f16')]
    for name,source in [('rokoko','rokoko.cpp'),('rokoko.fp16','rokoko_f16.cpp')]:
        with tempfile.TemporaryDirectory(prefix='rokoko-mutation-') as d:
            d=Path(d);text=(ROOT/'src'/source).read_text();needle='const int true_L=L;';assert text.count(needle)==1
            mutated=d/'model.cpp';mutated.write_text(text.replace(needle,needle+'\n    L=((L+31)/32)*32; // isolated historical padding fault'))
            binary=d/'mutant';cmd=[os.environ.get('CXX','g++'),'-std=c++17','-O2','-mavx2','-mfma','-I'+str(cuda/'include'),'-I'+str(ROOT/'src'),
                str(ROOT/'tests/runtime.o'),str(mutated),str(ROOT/'src/weights.cpp'),*map(str,objs),'-L'+str(cuda/'lib64'),'-lcudart','-lpthread','-o',str(binary)]
            subprocess.run(cmd,check=True,capture_output=True)
            p=paths(args,ROOT/name)
            proc=subprocess.run([str(binary),str(p['weights']),str(p['g2p']),str(p['voices']),str(d/'out')],capture_output=True,text=True,timeout=120)
            assert proc.returncode==1 and 'true frame length assertion' in proc.stderr,(name,proc.returncode,proc.stderr)
            report['controls'][name]=dict(returncode=proc.returncode,stderr=proc.stderr.strip(),fault='production decode rounded to 32 frames')
    save_report(ROOT/'tests/results/mutations.json',report);print('PASS: both padding mutants reached and failed the intended frame-length assertion.')
if __name__=='__main__':main()
