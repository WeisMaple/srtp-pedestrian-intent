"""Download and verify the pinned release model bundle without credentials."""
import pathlib,json,urllib.request,hashlib,zipfile,shutil,tempfile
ROOT=pathlib.Path(__file__).resolve().parent

def main():
    spec=json.loads((ROOT/'docs/final/release_asset.json').read_text())
    dest=ROOT/'weights/final'
    manifest=json.loads((ROOT/'docs/final/manifest.json').read_text())
    def matches():
        return all((dest/n).is_file() and hashlib.sha256((dest/n).read_bytes()).hexdigest()==r['sha256'] for n,r in manifest.items())
    if dest.exists():
        if matches():print('Final model files already verified:',dest);return
        raise SystemExit('Existing weights/final differs from the pinned model. Preserve or rename it before installing.')
    (ROOT/'weights').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='srtp-download-') as tmp:
        folder=pathlib.Path(tmp);archive=folder/spec['file']
        with urllib.request.urlopen(spec['url'],timeout=120) as response,archive.open('wb') as output:shutil.copyfileobj(response,output)
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=spec['sha256']:raise RuntimeError('Release archive checksum mismatch')
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                target=(folder/entry.filename).resolve()
                if folder.resolve() not in target.parents:raise RuntimeError('Unsafe archive path')
            z.extractall(folder)
        shutil.copytree(folder/'final',dest)
    assert matches();print('Final model files verified:',dest)

if __name__=='__main__':main()
