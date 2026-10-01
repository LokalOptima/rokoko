"""Cache-aware local benchmark. Stores every output for checks outside timed requests."""
import argparse
import statistics
import json
from pathlib import Path
from support import ROOT, idle_gpu, model_args, paths, provenance, request, save_report, server, sha256, wav_info

TEXTS = ['Hello world.',
'The quick brown fox jumps over the lazy dog and then runs back home again through the meadow.',
'In the heart of every great city there lies a park, a green refuge from the concrete jungle that surrounds it. People come from all walks of life to sit beneath the ancient oak trees and watch the world go by. Children play on the swings while their parents chat on nearby benches, sharing stories of their day.']

def summary(rows):
    vals=sorted(r['wall_ms'] for r in rows)
    return dict(n=len(vals), median_ms=statistics.median(vals), p95_ms=vals[min(len(vals)-1, int(.95*len(vals)))],
                minimum_ms=vals[0], maximum_ms=vals[-1]) if vals else {}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary',type=Path,action='append')
    ap.add_argument('--repeats',type=int,default=5)
    ap.add_argument('--mixed',type=int,default=200)
    ap.add_argument('--output',type=Path,default=ROOT/'tests/results/bench.json')
    model_args(ap); args=ap.parse_args()
    if args.repeats < 1 or args.mixed < 0: ap.error('repeats must be positive; mixed must be nonnegative')
    idle_gpu()
    report=dict(kind='descriptive benchmark; no perceptual equivalence claim', workloads=[])
    mixed=(ROOT/'tests/frontend/plain_test.txt').read_text().splitlines()[:args.mixed]
    for binary in args.binary or [ROOT/'rokoko']:
        artifact_dir=args.output.parent/(args.output.stem+'-'+binary.name)
        artifact_dir.mkdir(parents=True,exist_ok=True)
        run=dict(provenance=provenance(binary,bundled=True), artifacts={k:sha256(v) for k,v in paths(args).items() if v.is_file()}, requests=[])
        with server(binary,args,artifact_dir/'server.log') as (base,startup):
            run['startup_to_health_ms']=startup
            workload=[]
            for i,text in enumerate(TEXTS):
                workload += [('first_request_'+str(i),text), *[('repeat_'+str(i),text) for _ in range(args.repeats)]]
            workload += [('mixed',text) for text in mixed]
            for i,(label,text) in enumerate(workload):
                status,headers,data,wall=request(base,'/synthesize',dict(text=text))
                if status != 200: raise RuntimeError(f'{label}: HTTP {status}: {data!r}')
                info=wav_info(data); file=artifact_dir/f'{i:04d}.wav'; file.write_bytes(data)
                row=dict(index=i,workload=label,text=text,voice='af_heart',wall_ms=wall,wav=str(file),**info,
                         telemetry={k.lower():v for k,v in headers.items() if k.lower().startswith('x-')})
                run['requests'].append(row)
                if i%25==0:
                    status,_,body,_=request(base,'/stats')
                    if status==200: run.setdefault('memory_series',[]).append(dict(request=i,**json.loads(body)))
                if i%25 == 0: print(f'{binary.name}: {i+1}/{len(workload)} requests',flush=True)
            # Telemetry is absent in old binaries; do not infer cache state from sentence identity.
            status,_,data,_=request(base,'/stats')
            if status == 200:
                run['final_stats']=json.loads(data)
        run['summary']={label:summary([r for r in run['requests'] if r['workload']==label])
                        for label in sorted({r['workload'] for r in run['requests']})}
        for state in ('hit','miss'):
            rows=[r for r in run['requests'] if int(r['telemetry'].get('x-decode-'+('hits' if state=='hit' else 'misses'),'0'))>0]
            run['summary']['observed_decode_'+state]=summary(rows)
        run['cache_observed']=any('x-encode-hits' in r['telemetry'] for r in run['requests'])
        report['workloads'].append(run); save_report(args.output,report)
        print(binary.name,run['summary'],flush=True)
    print(f'Saved {args.output}')

if __name__ == '__main__': main()
