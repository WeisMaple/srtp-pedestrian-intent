"""Released bundles install correctly without changing an existing installation."""
import contextlib,hashlib,io,json,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
import download_final_models as downloader

class ModelDownloadTests(unittest.TestCase):
    def test_current_and_historical_archive_layouts(self):
        for prefix in ['weights/final','final']:
            with self.subTest(prefix=prefix), tempfile.TemporaryDirectory() as temp:
                root=Path(temp);(root/'docs/final').mkdir(parents=True)
                payload=b'fixture-model';manifest={'stgcn.pt':{'sha256':hashlib.sha256(payload).hexdigest()}}
                archive=io.BytesIO()
                with zipfile.ZipFile(archive,'w') as z:z.writestr(prefix+'/stgcn.pt',payload)
                content=archive.getvalue()
                (root/'docs/final/manifest.json').write_text(json.dumps(manifest))
                (root/'docs/final/release_asset.json').write_text(json.dumps({'file':'model.zip','url':'https://example.invalid/model.zip','sha256':hashlib.sha256(content).hexdigest()}))
                with patch.object(downloader,'ROOT',root),patch.object(downloader.urllib.request,'urlopen',return_value=io.BytesIO(content)) as fetch,contextlib.redirect_stdout(io.StringIO()):
                    downloader.main();self.assertEqual((root/'weights/final/stgcn.pt').read_bytes(),payload)
                    downloader.main();self.assertEqual(fetch.call_count,1)
                    (root/'weights/final/stgcn.pt').write_bytes(b'existing-modified')
                    with self.assertRaises(SystemExit):downloader.main()
                    self.assertEqual((root/'weights/final/stgcn.pt').read_bytes(),b'existing-modified')
