"""Verify expected final words in the last synthesized chunk, not just duration."""
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
from quality import wav
from support import ROOT, save_report, sha256
from scoring import words

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--asr',type=Path,required=True);ap.add_argument('--asr-weights',type=Path,required=True);args=ap.parse_args()
    rows=[]
    for case in ('5000','20000','no_sentence_delimiters'):
        root=ROOT/'tests/results/rokoko'/case
        chunks=sorted(root.glob('*/audio.f32'),key=lambda p:int(p.parent.name));assert chunks, 'run test-gpu first'
        output=root/'last.wav';wav(output,np.fromfile(chunks[-1],dtype='<f4'));rows.append(dict(binary='rokoko',case=case,wav=str(output)))
    p=subprocess.run([str(args.asr.resolve()),'--weights',str(args.asr_weights.resolve()),*[r['wav'] for r in rows]],capture_output=True,text=True,check=True,timeout=180)
    hypotheses=p.stdout.splitlines();assert len(hypotheses)==len(rows)
    for row,hyp in zip(rows,hypotheses):
        row['transcript']=hyp
        assert words(hyp)[-4:]==['final','words','purple','lantern'],row
        expected=['final','words','purple','lantern']
        assert words(hyp)[-4:]==expected and words(hyp)[:-1][-4:]!=expected, 'omitted-tail control'
    save_report(ROOT/'tests/results/long-tail.json',dict(asr_sha256=sha256(args.asr),weights_sha256=sha256(args.asr_weights),cases=rows))
    print('PASS expected final words in all three long-input tail clips.')
if __name__=='__main__':main()
