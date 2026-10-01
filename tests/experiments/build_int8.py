"""Build an isolated INT8 vocoder experiment from an already-built CPU checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cpu-build', type=Path, default=ROOT/'build/cpu')
    ap.add_argument('--output', type=Path, default=ROOT/'build/int8')
    args = ap.parse_args(); build = args.cpu_build.resolve(); out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    includes = [ROOT/'src', ROOT/'tests/experiments',
                build/'openblas/src/openblas_build', build/'onednn/src/onednn_build/include',
                build/'onednn/src/onednn_build-build/include']
    flags = ['c++', '-O3', '-DNDEBUG', '-std=c++17', '-DROKOKO_CPU=1',
             '-mavx2', '-mfma', '-mf16c', '-fno-math-errno', *['-I'+str(p) for p in includes]]
    source = (ROOT/'src/rokoko.cpp').read_text()
    marker = 'static void adain_resblock1_forward('
    assert source.count(marker) == 1
    position = source.index('{', source.index(marker)) + 1
    source = '#include "int8.h"\n' + source[:position] + '\n    cpu::Int8Scope int8_scope;\n' + source[position:]
    (out/'rokoko.cpp').write_text(source)
    source = (ROOT/'src/cpu/convolution.cpp').read_text()
    for a, b in [('void convolution(', 'void fp32_convolution('),
                 ('void clear_convolutions()', 'void fp32_clear_convolutions()')]:
        assert source.count(a) == 1
        source = source.replace(a, b)
    for header in ('convolution.h', 'math.h', 'parallel.h'):
        source = source.replace('#include "'+header+'"', '#include "cpu/'+header+'"')
    source += '\n#include "int8_conv.inc"\n'
    (out/'convolution.cpp').write_text(source)
    objects = []
    for name in ('rokoko', 'convolution'):
        obj = out/(name+'.o'); objects.append(str(obj))
        subprocess.run([*flags, '-c', str(out/(name+'.cpp')), '-o', str(obj)], check=True)
    libraries = [build/'librokoko_lib.a', build/'onednn/src/onednn_build-build/src/libdnnl.a',
                 build/'openblas/src/openblas_build/libopenblasrokoko_haswellp-r0.3.30.a']
    for name, main in [('rokoko.int8', 'CMakeFiles/rokoko.dir/src/main.cu.o'),
                       ('runtime', 'CMakeFiles/rokoko_runtime.dir/tests/runtime.cu.o')]:
        subprocess.run(['c++', '-O3', str(build/main), *objects, *map(str,libraries),
                        '-lm', '-lpthread', '-ldl', '-o', str(out/name)], check=True)
    subprocess.run([*flags, str(ROOT/'tests/experiments/int8_ops.cpp'), objects[1],
                    *map(str,libraries), '-lm', '-lpthread', '-ldl',
                    '-o', str(out/'operators')], check=True)
    sources = [ROOT/'src/rokoko.cpp', ROOT/'src/cpu/convolution.cpp', *sorted((ROOT/'tests/experiments').glob('*.*'))]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    (out/'build.json').write_text(json.dumps(dict(
        experiment='INT8 vocoder residual convolutions; dynamic per-tensor activations, per-output-channel weights',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
        libraries={str(p):sha(p) for p in libraries},
        binaries={name:sha(out/name) for name in ('rokoko.int8','runtime')}),indent=2)+'\n')
    print('Built',out/'rokoko.int8',flush=True)

if __name__ == '__main__': main()
