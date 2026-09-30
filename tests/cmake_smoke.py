"""Build and synthesize with both CMake configurations; build success alone is insufficient."""
import argparse
import subprocess
from pathlib import Path
from support import ROOT, model_args, model_flags, provenance, save_report, wav_info, idle_gpu

def main():
    ap=argparse.ArgumentParser(description=__doc__);model_args(ap)
    ap.add_argument('--fp16-build',type=Path,default=ROOT/'tests/results/cmake-fp16')
    ap.add_argument('--fp32-build',type=Path,default=ROOT/'tests/results/cmake-fp32')
    ap.add_argument('--no-build',action='store_true',help='test already-built directories')
    args=ap.parse_args();idle_gpu();report={}
    for variant,build in [('rokoko.fp16',args.fp16_build),('rokoko',args.fp32_build)]:
        out=ROOT/'tests/results'/('cmake-'+variant);out.mkdir(parents=True,exist_ok=True)
        with (out/'build.log').open('wb') as log:
            if not args.no_build:
                subprocess.run(['cmake','-S',str(ROOT),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release',
                    '-DROKOKO_FP16='+('ON' if variant.endswith('fp16') else 'OFF')],stdout=log,stderr=log,check=True)
                subprocess.run(['cmake','--build',str(build),'-j2'],stdout=log,stderr=log,check=True)
        binary=build/'rokoko';output=out/'smoke.wav'
        with (out/'synthesis.log').open('wb') as log:
            subprocess.run([str(binary),'Hello world.',*model_flags(args,ROOT/variant),'-o',str(output)],stdout=log,stderr=log,check=True,timeout=90)
        report[variant]=dict(provenance=provenance(binary),wav=wav_info(output.read_bytes()),
            architecture=[x for x in (build/'CMakeCache.txt').read_text().splitlines() if x.startswith('CMAKE_CUDA_ARCHITECTURES:')])
    save_report(ROOT/'tests/results/cmake.json',report);print('PASS both CMake executables synthesize valid audio.')
if __name__=='__main__':main()
