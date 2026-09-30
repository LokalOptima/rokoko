"""Offline CPU regressions. Expected vocabulary comes from official config.json."""
import json
import io
import struct
import wave
from scoring import score, paired_interval
import random
import re
import subprocess
import tempfile
import os
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
VOCAB = json.loads((HERE / "fixtures/vocab.json").read_text())["vocab"]

def helper(mode, text, *args):
    return subprocess.run([str(HERE / "helpers"), mode, *map(str,args)], input=text, text=True,
                          capture_output=True, check=True, timeout=10).stdout

def chunks(text):
    return [(int(a), int(b), bytes.fromhex(value).decode())
            for a,b,value in (line.split() for line in helper("chunks", text).splitlines())]

def assert_coverage(test, source, pieces, limit=510):
    raw=source.encode(); previous=0
    for begin,end,piece in pieces:
        test.assertTrue(previous <= begin < end <= len(raw), "chunk spans overlap or exceed source")
        test.assertFalse(raw[previous:begin].decode().strip(), "chunk content lost")
        test.assertEqual(raw[begin:end].decode(),piece,"chunk content changed")
        test.assertEqual(piece,piece.strip(),"untrimmed chunk")
        test.assertLessEqual(len(piece),limit,"chunk exceeds model limit")
        previous=end
    test.assertFalse(raw[previous:].decode().strip(), "chunk content lost")

def canon_text(s):
    return " ".join(re.sub(r'[.,!?;\"]', ' ', s.lower()).split())

def normalizer_rows():
    result = []
    for line in (HERE / "frontend/norm_test.tsv").read_text().splitlines():
        if not line.strip() or line.startswith(("#", "class\t")): continue
        kind, raw, accepted = line.split("\t")[:3]
        result.append((kind, raw, accepted.split("||")))
    return result

def assert_normalized(test, rows, actual):
    test.assertEqual(len(rows), len(actual), "normalizer output count")
    for (kind, raw, accepted), value in zip(rows, actual):
        test.assertIn(canon_text(value), [canon_text(x) for x in accepted],
                      f"normalizer {kind}: {raw!r}")

class CPUChecks(unittest.TestCase):
    def test_pcm16_transport(self):
        data=subprocess.check_output([str(HERE/'audio')])
        with wave.open(io.BytesIO(data)) as f:
            self.assertEqual((f.getnchannels(),f.getsampwidth(),f.getframerate(),f.getnframes()),(1,2,24000,9))
            self.assertEqual(struct.unpack('<9h',f.readframes(9)),(-32767,-32767,-16383,0,0,0,16383,32767,32767))
        self.assertEqual(len(data),62)

    def test_scoring_controls(self):
        self.assertEqual(score('Do not go.','DO NOT GO!'),dict(errors=0,words=3))
        for ref,hyp in [('do not go','do go'),('five birds','six birds'),('final purple lantern','final')]:
            self.assertGreater(score(ref,hyp)['errors'],0)
        self.assertEqual(paired_interval([dict(errors=1,words=2)]*3,[dict(errors=0,words=2)]*3,draws=100),[.5,.5])

    def test_official_vocabulary(self):
        actual = {chr(int(cp)): int(token) for cp, token in
                  (line.split() for line in helper("vocab", "").splitlines())}
        self.assertEqual(actual, VOCAB)

    def test_tokenization(self):
        cases = ["", "həlˈoʊ wˈɜːld.", "a☃b", "☃", "—…", "".join(VOCAB)]
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(list(map(int, helper("tokens", text).split())),
                                 [0] + [VOCAB[c] for c in text if c in VOCAB] + [0])

    def test_chunk_boundaries_and_content(self):
        for n in (1, 509, 510, 511, 1021, 5000):
            for text in ("a"*n, "ə"*n, ("həlˈoʊ, wɜːld! "*400)[:n]):
                with self.subTest(length=n, text=text[:20]):
                    assert_coverage(self, text, chunks(text))
        rng = random.Random(9401)
        for _ in range(20):
            text = "".join(rng.choice("ab əˈ.!?,:;—…") for _ in range(rng.randrange(1, 2200)))
            assert_coverage(self, text, chunks(text))

    def test_chunk_oracle_rejects_loss_and_oversize(self):
        source = "a" * 511
        with self.assertRaisesRegex(AssertionError, "content lost"):
            assert_coverage(self, source, [(0,510,"a"*510)])
        with self.assertRaisesRegex(AssertionError, "exceeds model limit"):
            assert_coverage(self, source, [(0,511,source)])

    def test_empty_and_whitespace(self):
        for text in ("", "  ", "\t\r\n", "\u2003\u00a0"):
            self.assertEqual(chunks(text), [])

    def test_g2p_text_spans(self):
        for size in (1,2047,2048,2049,5000,20000):
            for text in ("a"*size, ("Hello world. "*2000)[:size], "ə"*size):
                spans=[tuple(map(int,line.split())) for line in helper("split",text,2048).splitlines()]
                raw=text.encode()
                assert_coverage(self,text,[(a,b,raw[a:b].decode()) for a,b in spans],2048)

    def test_style_length_before_filtering(self):
        for text in ("a", "ab", "a☃b", "ə"*509, "ə"*510):
            self.assertEqual(int(helper("style",text)),len(text)-1)
        for text in ("", "a"*511):
            with self.assertRaises(subprocess.CalledProcessError): helper("style",text)

    def test_malformed_utf8(self):
        for value in (b'\xc0\xaf', b'\xed\xa0\x80', b'\xf4\x90\x80\x80', b'\x80', b'\xe2\x82'):
            p=subprocess.run([str(HERE/"helpers"),"chunks"],input=value,capture_output=True)
            self.assertEqual(p.returncode,2)
            self.assertIn(b'invalid UTF-8',p.stderr)

    def test_style_source_mutation(self):
        # Compile the real helper with precisely the historical off-by-one restored.
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            source=(HERE.parent/'src/phonemes.h').read_text()
            self.assertIn('return n - 1;',source)
            (d/'phonemes.h').write_text(source.replace('return n - 1;', 'return n;'))
            subprocess.run([os.environ.get('CXX','g++'),'-std=c++17','-I'+str(d),
                            str(HERE/'helpers.cpp'),'-o',str(d/'mutant')],check=True,capture_output=True)
            out=subprocess.check_output([str(d/'mutant'),'style'],input=b'ab')
            with self.assertRaises(AssertionError): self.assertEqual(int(out),1)

    def test_normalizer_handwritten(self):
        rows = normalizer_rows()
        result = subprocess.run([str(HERE / "frontend/normalize_cli")],
            input="\n".join(r[1] for r in rows)+"\n", text=True,
            capture_output=True, check=True, timeout=30)
        assert_normalized(self, rows, result.stdout.splitlines())

    def test_normalizer_rejects_identity(self):
        rows = normalizer_rows()
        with self.assertRaisesRegex(AssertionError, "normalizer"):
            assert_normalized(self, rows, [r[1] for r in rows])

    def test_reviewed_snapshot(self):
        rows = [line.split("\t") for line in
                (HERE / "frontend/real_world.reviewed.tsv").read_text().splitlines()[1:]]
        result = subprocess.run([str(HERE / "frontend/normalize_cli")],
            input="\n".join(r[0] for r in rows)+"\n", text=True,
            capture_output=True, check=True, timeout=30)
        self.assertEqual(result.stdout.splitlines(), [r[1] for r in rows])

if __name__ == "__main__":
    print("CPU-only checks; GPU, artifacts and audio evaluation are separate.", flush=True)
    unittest.main()
