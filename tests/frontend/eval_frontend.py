# /// script
# requires-python = ">=3.12"
# dependencies = ["misaki[en]==0.9.4", "espeakng-loader>=0.2.4", "phonemizer-fork>=3.3.2", "pip"]
# ///
"""Evaluate rokoko's text frontend (normalizer + C++ G2P) on data that was
written independently of the code. See README.md in this directory.

    make test-frontend                      # evaluates ~/.cache/rokoko/g2p.bin
    make test-frontend G2P=weights/g2p.bin  # any other G2P model

Exit code is non-zero if any hard check fails.
"""
import argparse, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(binary, lines, *args):
    out = subprocess.run([str(binary), *args], input="\n".join(lines) + "\n",
                         capture_output=True, text=True, check=True).stdout.split("\n")
    assert len(out) >= len(lines), f"{binary.name}: {len(out)} outputs for {len(lines)} lines"
    return out[: len(lines)]


def canon_text(s):
    # Sentence punctuation doesn't matter; ':' '-' '/' do (the G2P reads them)
    return " ".join(re.sub(r"[.,!?;\"]", " ", s.lower()).split())


def lenient(ph):
    ph = ph.replace("ˈ", "").replace("ˌ", "").replace("ᵊ", "ə")
    ph = re.sub(r'[.,;:!?"—–‘’“”()\[\]{}]', "", ph)
    return " ".join(ph.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g2p", default=str(Path.home() / ".cache/rokoko/g2p.bin"))
    a = ap.parse_args()
    norm_bin, g2p_bin = HERE / "normalize_cli", HERE / "g2p_check"

    from misaki import en, espeak  # same configuration the G2P training data used
    misaki = en.G2P(trf=False, british=False, fallback=espeak.EspeakFallback(british=False), unk="")
    ref = lambda t: misaki(t)[0] or ""

    rows = []
    for line in open(HERE / "norm_test.tsv"):
        if line.startswith(("#", "class\t")) or not line.strip():
            continue
        cls, raw, acc = line.rstrip("\n").split("\t")[:3]
        rows.append((cls, raw, [x.strip() for x in acc.split("||")]))
    plain = [l.strip() for l in open(HERE / "plain_test.txt") if l.strip()]
    failed = []

    # 1. Normalizer vs hand-written spoken forms
    normed = run(norm_bin, [r[1] for r in rows])
    bad = [(r, n) for r, n in zip(rows, normed) if canon_text(n) not in {canon_text(x) for x in r[2]}]
    print(f"normalizer, hand-written set:  {len(rows) - len(bad)}/{len(rows)}")
    for (cls, raw, acc), n in bad:
        print(f"    FAIL {cls}: {raw!r} -> {n!r}   expected: {' || '.join(acc)}")
    if bad:
        failed.append("normalizer")

    # 2. G2P alone on plain sentences, vs misaki
    plain_normed = run(norm_bin, plain)
    pred = run(g2p_bin, plain_normed, a.g2p)
    refs = [ref(s) for s in plain]
    ok = sum(lenient(p) == lenient(r) for p, r in zip(pred, refs))
    strict = sum(p.strip() == r.strip() for p, r in zip(pred, refs))
    print(f"G2P exact, stress retained:     {strict}/{len(plain)} (diagnostic, no new threshold)")
    print(f"G2P, plain sentences:           {ok}/{len(plain)} ({100 * ok / len(plain):.1f}%)")
    if ok / len(plain) < 0.985:
        failed.append("g2p plain")

    # 3. End to end: raw text -> normalizer -> G2P, vs misaki on any accepted form
    pred = run(g2p_bin, normed, a.g2p)
    e2e_bad = [(r, n, p) for r, n, p in zip(rows, normed, pred)
               if lenient(p) not in {lenient(ref(x)) for x in r[2]}]
    print(f"end to end, hand-written set:  {len(rows) - len(e2e_bad)}/{len(rows)}")
    for (cls, raw, acc), n, p in e2e_bad:
        print(f"    miss {cls}: {n!r} -> {p}   ref: {ref(acc[0])}")
    if (len(rows) - len(e2e_bad)) / len(rows) < 0.97:
        failed.append("end to end")

    # 4. Real-world sentences: compare to the reviewed snapshot. A change is not
    #    necessarily wrong, but it has to be looked at by a person.
    snap = [l.rstrip("\n").split("\t") for l in open(HERE / "real_world.reviewed.tsv")][1:]
    now = run(norm_bin, [s[0] for s in snap])
    changed = [(s, n) for s, n in zip(snap, now) if n != s[1]]
    known = sum(1 for s in snap if s[2] != "ok")
    print(f"real-world snapshot:            {len(snap) - len(changed)}/{len(snap)} unchanged "
          f"({known} known issues listed in real_world.reviewed.tsv)")
    for s, n in changed:
        print(f"    CHANGED {s[0]!r}\n       was: {s[1]!r}\n       now: {n!r}")
    if changed:
        failed.append("real-world snapshot (review the changes, then update real_world.reviewed.tsv)")

    print("\nOK" if not failed else "\nFAILED: " + ", ".join(failed))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
