"""Check the consolidated training/runtime boundary with the actual V11 checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training/g2p'))
from normalizer import Normalizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from train import load_model_from_ckpt, ctc_greedy
    torch.set_num_threads(2)

    normalizer = Normalizer()
    cases = ['Hello world.', '', 'I have $5.', 'The year was 1969.', '']
    actual = normalizer.normalize_batch(cases)
    expected = subprocess.check_output([normalizer.binary_path],
                                      input='\n'.join(cases) + '\n', text=True).split('\n')[:-1]
    assert actual == expected and len(actual) == len(cases)
    assert normalizer.normalize_batch([]) == []
    for invalid in ['two\nlines', 'two\tcolumns', 'carriage\rreturn']:
        try:
            normalizer.normalize(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('accepted a broken training record')

    with tempfile.TemporaryDirectory(prefix='rokoko-training-') as tmp:
        tmp = Path(tmp)
        export = tmp / 'g2p.bin'
        subprocess.run([sys.executable, str(ROOT/'training/g2p/train.py'), 'export',
                        '--checkpoint', str(args.checkpoint), '--output', str(export)], check=True)
        approved = json.loads((ROOT/'tests/fixtures/artifacts.json').read_text())['files']['g2p.bin']
        assert hashlib.sha256(export.read_bytes()).hexdigest() == approved['sha256'], 'V11 export changed'

        # Python checkpoint and native runtime agree on inputs normalized by current production code.
        raw = (ROOT/'tests/frontend/plain_test.txt').read_text().splitlines()[:12]
        texts = normalizer.normalize_batch(raw)
        native = subprocess.check_output([str(ROOT/'tests/frontend/g2p_check'), str(export)],
                                         input='\n'.join(texts)+'\n', text=True).splitlines()
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model, chars, phones, cfg = load_model_from_ckpt(checkpoint)
        predictions = []
        with torch.no_grad():
            for text in texts:
                result = model(torch.tensor([chars.encode(text)]), torch.tensor([len(text)]))
                logits = result[0] if isinstance(result, tuple) else result
                predictions.append(phones.decode(ctc_greedy(logits[0, :len(text)*cfg['up']])))
        assert native == predictions, 'checkpoint/native G2P mismatch'

        # Exercise data preparation through the actual shared normalizer and phonemizer.
        raw_path = tmp/'raw.txt'
        raw_path.write_text('Hello world.\nHello world.\nI have $5.\n')
        dataset = tmp/'prepared.tsv'
        command = [sys.executable, str(ROOT/'training/g2p/prepare_data.py'),
                   '--input', str(raw_path), '--output', str(dataset), '--workers', '1']
        subprocess.run(command, check=True)
        rows = [line.split('\t') for line in dataset.read_text().splitlines()]
        assert [row[0] for row in rows] == normalizer.normalize_batch(['Hello world.', 'I have $5.'])
        assert all(len(row)==2 and row[1] for row in rows)
        manifest = json.loads(Path(str(dataset)+'.manifest.json').read_text())
        assert manifest['output']['sha256'] == hashlib.sha256(dataset.read_bytes()).hexdigest()
        before = dataset.read_bytes()
        refused = subprocess.run(command, capture_output=True, text=True)
        assert refused.returncode != 0 and 'already exists' in refused.stderr
        assert dataset.read_bytes() == before
    print('PASS: shared normalizer, exact V11 export, checkpoint/native agreement, data preparation and provenance.')


if __name__ == '__main__':
    main()
