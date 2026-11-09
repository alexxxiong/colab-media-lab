"""Verify the archive can recover immutable results and preserve their hashes."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile


class EvidencePackageTests(unittest.TestCase):
    def test_chunked_roundtrip_and_private_log_exclusion(self):
        source=Path(__file__).with_name('worker.py')
        tree=ast.parse(source.read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='package_results')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'remote'
            root.mkdir()
            (root/'results').mkdir()
            (root/'results'/'large.mp4').write_bytes(b'v'*(63*1024**2))
            (root/'results'/'small.mp4').write_bytes(b's'*(2*1024**2))
            (root/'startup-resources.jsonl').write_text('{"at":1}\n')
            (root/'mount.log').write_text('PRIVATE OAUTH MATERIAL')
            def sha(path):
                with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
            def write_json(path,value): path.write_text(json.dumps(value))
            namespace=dict(ROOT=root,SESSION='test',Path=Path,shutil=shutil,zipfile=zipfile,
                           event=lambda *a,**k:None,state=lambda *a,**k:None,
                           publish=lambda *a:None,sha=sha,write_json=write_json)
            exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
            namespace['package_results']({'id':'90-evidence'})
            manifest=json.loads((root/'bundles'/'90-evidence-manifest.json').read_text())
            self.assertEqual(len(manifest['chunks']),2)
            self.assertNotIn('mount.log',manifest['files'])
            self.assertIn('startup-resources.jsonl',manifest['files'])
            recovered=Path(temporary)/'recovered'
            for chunk in manifest['chunks']:
                archive=root/'bundles'/chunk['name']
                self.assertEqual(sha(archive),chunk['sha256'])
                with zipfile.ZipFile(archive) as stream: stream.extractall(recovered)
            for name,record in manifest['files'].items():
                self.assertEqual(sha(recovered/name),record['sha256'])
                self.assertEqual((recovered/name).stat().st_size,record['bytes'])
            self.assertTrue(json.loads((root/'results'/'90-evidence-command.json').read_text())['completed'])


if __name__=='__main__': unittest.main()
