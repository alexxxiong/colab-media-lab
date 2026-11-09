"""CPU-only data, privacy, link and integrity checks. Never calls a cloud API."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
PATTERNS={
    'email':r'(?<![\w.+-])[\w.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
    'personal_windows_path':r'[A-Za-z]:[\\/]+Users[\\/]+[^\s\x22\x27]+',
    'personal_linux_path':r'/(?:home|mnt/c/Users)/[^/\s\x22\x27]+',
    'private_drive_link':r'https?://(?:drive|docs)\.google\.com/',
    'google_oauth_access':r'ya29\.[A-Za-z0-9._-]{20,}',
    'google_oauth_refresh':r'1//[A-Za-z0-9_-]{30,}',
    'github_token':r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})',
    'api_key':r'AIza[A-Za-z0-9_-]{30,}',
    'private_key':r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'oauth_state_url':r'https?://[^\s\x22\x27]+[?&]state=[^\s\x22\x27]+',
    'runtime_proxy_value':r'colab-runtime-proxy-token=[A-Za-z0-9._~-]{12,}',
    'runtime_ip_url':r'https?://(?:\d{1,3}\.){3}\d{1,3}',
}
BLOCKED_SUFFIXES=('.zip','.tar.gz','.safetensors','.pt','.pth','.so','.pem','.key','.log')
BLOCKED_NAMES={'token.json','sessions.json','.env','mount.log','storage-config.json','current-session.json'}


def findings(text):
    return [name for name,pattern in PATTERNS.items() if re.search(pattern,text)]


def digest(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def public_files():
    return [p for p in ROOT.rglob('*') if p.is_file() and not any(part in ('.git','.venv','__pycache__') for part in p.relative_to(ROOT).parts)]


def validate(include_media_metadata=False):
    failures=[]
    for path in public_files():
        relative=path.relative_to(ROOT).as_posix()
        if path.name in BLOCKED_NAMES or path.name.endswith(BLOCKED_SUFFIXES): failures.append((relative,'blocked artifact'))
        # ASCII secrets and identifiers are also detectable inside uncompressed
        # media metadata; compressed metadata is separately inspected before release.
        content=path.read_bytes().decode('utf-8',errors='ignore')
        binary=path.suffix.lower() in ('.png','.mp4')
        for kind in findings(content):
            # Random compressed image/video bytes are not prose. They can
            # accidentally resemble email or paths when decoded with ignore.
            if not binary or kind in ('google_oauth_access','google_oauth_refresh','github_token','api_key','private_key'):
                failures.append((relative,kind))
        if binary and include_media_metadata:
            if path.suffix.lower()=='.png':
                from PIL import Image
                with Image.open(path) as image:
                    metadata={k:v for k,v in image.info.items() if isinstance(v,(str,int,float))}
                    metadata.update({str(k):v for k,v in image.getexif().items() if isinstance(v,(str,int,float))})
            else:
                import av
                with av.open(str(path)) as container:
                    metadata={'container':dict(container.metadata),'streams':[dict(s.metadata) for s in container.streams]}
            for kind in findings(json.dumps(metadata,ensure_ascii=False)):
                failures.append((relative,'metadata '+kind))
    result=json.loads((ROOT/'data/results.json').read_text())
    assert result['accepted_main_clips']==0 and result['formal_matrix_completed'] is False
    assert result['formal_cold_starts']==0 and result['formal_hot_runs']==0
    assert result['colab_text_to_image_validated'] is False
    assert result['large_scale_batch_or_concurrency_validated'] is False
    with (ROOT/'data/observations.csv').open(newline='') as stream: rows=list(csv.DictReader(stream))
    assert len(rows)==7
    assert sum(r['technical_passed']=='True' for r in rows)==4
    for row in rows:
        if row['generation_seconds']:
            expected=float(row['generation_seconds'])*10.05/3600
            assert abs(expected-float(row['generation_cu_estimate']))<1e-9
    for item in json.loads((ROOT/'media/manifest.json').read_text()):
        assert digest(ROOT/item['path'])==item['sha256'],item['path']
        assert (ROOT/item['path']).stat().st_size==item['bytes']
    assert digest(ROOT/'media/first-frame.png')==result['first_frame']['sha256']
    for path in ROOT.rglob('*.md'):
        if '.git' in path.parts: continue
        text=path.read_text()
        for target in re.findall(r'\]\(([^)]+)\)',text):
            target=target.split('#')[0]
            if not target or re.match(r'^[a-z]+:',target): continue
            if not (path.parent/target).exists(): failures.append((str(path.relative_to(ROOT)),'missing link: '+target))
    if failures:
        for filename,kind in failures: print(filename+': '+kind)
        raise SystemExit('Publication validation failed')
    return public_files()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--write-manifest',action='store_true')
    parser.add_argument('--strict-manifest',action='store_true')
    parser.add_argument('--media-metadata',action='store_true')
    args=parser.parse_args()
    files=validate(include_media_metadata=args.media_metadata)
    manifest=ROOT/'PUBLICATION-MANIFEST.json'
    if args.write_manifest:
        manifest.write_text(json.dumps({'files':[{ 'path':p.relative_to(ROOT).as_posix(),
            'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(files) if p!=manifest],
            'gpu_started_for_publication':False},indent=2)+'\n')
    if args.strict_manifest:
        for item in json.loads(manifest.read_text())['files']:
            assert digest(ROOT/item['path'])==item['sha256'],item['path']
    print(json.dumps({'privacy_and_data_checks':'passed','files_checked':len(files),
                      'strict_manifest':args.strict_manifest,'media_metadata_checked':args.media_metadata,'gpu_started':False}))


if __name__=='__main__': main()
