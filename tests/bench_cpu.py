"""Alternate preserved and current CPU binaries; retain timings and every WAV."""
import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import time

from bench import TEXTS, summary
from support import embedded_identity, provenance, request, save_report, server, sha256, wav_info


def cpu_snapshot():
    # Linux process counters include all BLAS threads. Subtract this driver and
    # its two server children from total CPU use to detect competing work.
    fields = list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
    busy = sum(fields) - fields[3] - fields[4]  # exclude idle and I/O wait
    pid = os.getpid()
    children = Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
    own = 0
    for process in [str(pid), *children]:
        try:
            fields = Path(f'/proc/{process}/stat').read_text().rsplit(')', 1)[1].split()
            own += int(fields[11]) + int(fields[12])
        except FileNotFoundError:
            pass
    return time.monotonic(), busy, own


def competing_cores(before, after):
    elapsed = after[0] - before[0]
    ticks = (after[1] - before[1]) - (after[2] - before[2])
    return max(0., ticks / os.sysconf('SC_CLK_TCK') / elapsed)


def wait_quiet(limit):
    attempts = 0
    while True:
        before = cpu_snapshot()
        time.sleep(1)
        cores = competing_cores(before, cpu_snapshot())
        if cores <= limit:
            return
        if attempts % 15 == 0:
            print(f'Waiting for competing CPU work ({cores:.1f} cores busy)', flush=True)
        attempts += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--before', type=Path, required=True)
    ap.add_argument('--after', type=Path, required=True)
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--after-timeout', type=int, choices=range(4, 31))
    ap.add_argument('--max-competing-cores', type=float, default=.5,
                    help='wait/retry when other processes exceed this CPU use (default: 0.5 core)')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.repeats < 1 or not 1 <= args.threads <= 64:
        ap.error('repeats must be positive; threads must be in 1..64')
    if not 0 < args.max_competing_cores < float('inf'):
        ap.error('max-competing-cores must be positive and finite')
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(method='alternating CPU requests, first request per text excluded; '
                         '200 ms idle gap before each request lets BLAS workers sleep',
                  current_source=provenance(), binaries={}, requests=[], summary=[])
    report['max_competing_cores'] = args.max_competing_cores
    with ExitStack() as stack:
        endpoints = {}
        for label, binary in [('before', args.before), ('after', args.after)]:
            identity = embedded_identity(binary)
            assert identity['backend'] == 'cpu', 'CPU builds required'
            if label == 'after':
                assert identity['files'] == report['binaries']['before']['assets']['files']
            env = dict(os.environ, ROKOKO_CPU_THREADS=str(args.threads))
            env.pop('OPENBLAS_THREAD_TIMEOUT', None)
            if label == 'after' and args.after_timeout is not None:
                env['OPENBLAS_THREAD_TIMEOUT'] = str(args.after_timeout)
            directory = args.output / label
            directory.mkdir(exist_ok=True)
            base, startup = stack.enter_context(server(binary, None, directory/'server.log', env=env))
            endpoints[label] = base
            report['binaries'][label] = dict(path=str(binary.resolve()), sha256=sha256(binary),
                assets=identity, threads=args.threads, startup_ms=startup,
                timeout=env.get('OPENBLAS_THREAD_TIMEOUT', 'build default'))
        for text_index, text in enumerate(TEXTS):
            repeat = -1
            attempt = 0
            while repeat < args.repeats:
                wait_quiet(args.max_competing_cores)
                order = ['before', 'after'] if repeat % 2 == 0 else ['after', 'before']
                pair = []
                for label in order:
                    time.sleep(.2)
                    snapshot = cpu_snapshot()
                    status, headers, data, wall = request(endpoints[label], '/synthesize', dict(text=text))
                    competing = competing_cores(snapshot, cpu_snapshot())
                    assert status == 200, (label, status, data)
                    wav = args.output/label/f'{text_index}-{attempt}.wav'
                    wav.write_bytes(data)
                    pair.append(dict(binary=label, text_index=text_index, text=text,
                        repeat=repeat, attempt=attempt, competing_cores=competing,
                        wall_ms=wall, wav=str(wav), **wav_info(data),
                        telemetry={k.lower(): v for k, v in headers.items() if k.lower().startswith('x-')}))
                accepted = all(r['competing_cores'] <= args.max_competing_cores for r in pair)
                for row in pair:
                    row['accepted'] = accepted
                report['requests'].extend(pair)
                save_report(args.output/'report.json', report)
                if accepted or repeat == -1:
                    repeat += 1
                else:
                    print(f'Retrying text {text_index}, repeat {repeat}: competing CPU work', flush=True)
                attempt += 1
            rows = [r for r in report['requests'] if r['text_index'] == text_index
                    and r['repeat'] >= 0 and r['accepted']]
            assert len({r['samples'] for r in rows}) == 1, 'audio length changed'
            result = dict(text=text, audio_seconds=rows[0]['audio_seconds'])
            for label in ('before', 'after'):
                result[label] = summary([r for r in rows if r['binary'] == label])
            result['time_reduction_pct'] = 100 * (1 - result['after']['median_ms']/result['before']['median_ms'])
            result['speedup'] = result['before']['median_ms']/result['after']['median_ms']
            report['summary'].append(result)
            save_report(args.output/'report.json', report)
            print(result, flush=True)


if __name__ == '__main__':
    main()
