"""Offline build preparation checks: bad inputs must never become trusted assets."""
import concurrent.futures
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import io

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('bundle_assets',ROOT/'scripts/bundle_assets.py')
bundle=importlib.util.module_from_spec(spec);spec.loader.exec_module(bundle)

class BundleTests(unittest.TestCase):
    def test_manifest_matches_independent_approval(self):
        build=json.loads(bundle.MANIFEST.read_text())['files']
        approved=json.loads((ROOT/'tests/fixtures/artifacts.json').read_text())['files']
        self.assertEqual({n:{k:v[k] for k in ('size','sha256')} for n,v in build.items()},approved)
        self.assertEqual(set(build),{'weights.fp16.bin','g2p.bin','voices/af_heart.bin'})

    def test_local_offline_reuse_and_corruption(self):
        payload=b'approved model bytes'
        meta=dict(size=len(payload),sha256=hashlib.sha256(payload).hexdigest(),url='https://unused.invalid/model')
        with tempfile.TemporaryDirectory() as d, patch.object(bundle.urllib.request,'urlopen',side_effect=AssertionError('network used')):
            d=Path(d);source=d/'source';source.mkdir();(source/'model').write_bytes(payload)
            build=d/'build';build.mkdir()
            with self.assertRaises(FileNotFoundError): bundle.prepare_one('model',meta,build,offline=True)
            # Concurrent builds must safely share an asset and leave no partial files.
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                files=list(pool.map(lambda _:bundle.prepare_one('model',meta,build,source,True),range(2)))
            self.assertEqual(files[0].read_bytes(),payload)
            stamp=files[0].stat().st_mtime_ns
            bundle.prepare_one('model',meta,build,offline=True)
            self.assertEqual(files[0].stat().st_mtime_ns,stamp)
            files[0].write_bytes(b'X'+payload[1:])
            with self.assertRaises(ValueError): bundle.prepare_one('model',meta,build,source,True)
            self.assertEqual((source/'model').read_bytes(),payload)
            self.assertEqual({p.name for p in build.iterdir()},{'model','.assets.lock'})

    def test_download_is_verified_before_promotion(self):
        payload=b'approved bytes'
        meta=dict(size=len(payload),sha256=hashlib.sha256(payload).hexdigest(),url='https://unused.invalid/model')
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            with patch.object(bundle.urllib.request,'urlopen',return_value=io.BytesIO(b'broken')):
                with self.assertRaises(ValueError): bundle.prepare_one('model',meta,d)
            self.assertFalse((d/'model').exists())
            self.assertEqual([p.name for p in d.iterdir()],['.assets.lock'])
            with patch.object(bundle.urllib.request,'urlopen',return_value=io.BytesIO(payload)):
                self.assertEqual(bundle.prepare_one('model',meta,d).read_bytes(),payload)

if __name__=='__main__': unittest.main()
