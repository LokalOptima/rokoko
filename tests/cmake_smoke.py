"""Build the standalone CMake executable and a library-only consumer."""
import argparse
import subprocess
from pathlib import Path
from support import ROOT, model_args, provenance, save_report, wav_info, idle_gpu
from bundle_smoke import check_binary


def main():
    ap=argparse.ArgumentParser(description=__doc__);model_args(ap)
    ap.add_argument('--build',type=Path)
    ap.add_argument('--backend',choices=('CUDA','CPU'),default='CUDA')
    ap.add_argument('--openblas-archive',type=Path)
    ap.add_argument('--no-build',action='store_true',help='test already-built directories')
    args=ap.parse_args()
    if args.backend=='CUDA': idle_gpu()
    report={};suffix='-cpu' if args.backend=='CPU' else ''
    build=args.build or ROOT/('build/cmake'+suffix)
    options=['-DCMAKE_BUILD_TYPE=Release','-DROKOKO_BACKEND='+args.backend,
             '-DROKOKO_ASSET_SOURCE='+str(args.models.resolve()),'-DROKOKO_OFFLINE=ON']
    if args.openblas_archive: options.append('-DROKOKO_OPENBLAS_URL='+str(args.openblas_archive.resolve()))
    out=ROOT/'tests/results'/('cmake'+suffix);out.mkdir(parents=True,exist_ok=True)
    with (out/'build.log').open('wb') as log:
        if not args.no_build:
            subprocess.run(['cmake','-S',str(ROOT),'-B',str(build),*options],stdout=log,stderr=log,check=True)
            subprocess.run(['cmake','--build',str(build),'-j2'],stdout=log,stderr=log,check=True)
    binary=build/'rokoko'
    report['rokoko']=dict(provenance=provenance(binary,bundled=True),standalone=check_binary(binary),
        architecture=[x for x in (build/'CMakeCache.txt').read_text().splitlines() if x.startswith('CMAKE_CUDA_ARCHITECTURES:')])
    # A separate project must get embedded data by linking only rokoko_lib.
    source=ROOT/('build/library-consumer'+suffix);source.mkdir(parents=True,exist_ok=True)
    main='main.cpp' if args.backend=='CPU' else 'main.cu'
    (source/main).write_text('''#include "rokoko.h"
int main(int argc,char** argv) {
    if (argc!=2) return 1;
    rokoko::TtsContext context;
    if (!context.init()) return 2;
    auto pipeline=context.pipeline();std::vector<float> audio;
    if (!pipeline.synthesize("Library-only consumer.",audio).empty()) return 3;
    return rokoko::write_wav(argv[1],audio.data(),audio.size(),24000) ? 0 : 4;
}
''')
    (source/'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.24)
set(CMAKE_CUDA_ARCHITECTURES native)
project(consumer LANGUAGES CXX '''+('' if args.backend=='CPU' else 'CUDA')+''')
set(CMAKE_CUDA_STANDARD 17)
set(ROKOKO_BUILD_LIB_ONLY ON CACHE BOOL "" FORCE)
add_subdirectory("'''+str(ROOT)+'''" rokoko)
add_executable(consumer '''+main+''')
target_link_libraries(consumer PRIVATE rokoko_lib)
''')
    with (source/'build.log').open('wb') as log:
        subprocess.run(['cmake','-S',str(source),'-B',str(source/'build'),*options],stdout=log,stderr=log,check=True)
        subprocess.run(['cmake','--build',str(source/'build'),'-j2'],stdout=log,stderr=log,check=True)
        subprocess.run([str(source/'build/consumer'),str(source/'smoke.wav')],stdout=log,stderr=log,check=True,timeout=90)
    report['library_consumer']=wav_info((source/'smoke.wav').read_bytes())
    save_report(ROOT/('tests/results/cmake'+suffix+'.json'),report)
    print('PASS standalone CMake executable and library-only consumer synthesize valid audio.')

if __name__=='__main__':main()
