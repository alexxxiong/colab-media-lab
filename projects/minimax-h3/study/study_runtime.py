"""Shared runtime staging and append-only evidence (runs only in Colab)."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor

ROOT=Path('/content/colab-lab/whale-study')
DRIVE=Path('/content/drive/MyDrive/Colab-Lab')
PROJECT=DRIVE/'notebooks/minimax-h3'
_options=json.loads((ROOT/'session-options.json').read_text()) if (ROOT/'session-options.json').is_file() else {}
ASSETS=DRIVE/'datasets/minimax-h3-benchmark'/_options.get('assets_version','whale-v1')
RUNS=DRIVE/'runs/minimax-h3-study/example-study'
_lock=threading.Lock()
_publish_lock=threading.Lock()

def write_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    temp.replace(path)

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def event(stage,**fields):
    item={'at':time.time(),'monotonic':time.monotonic(),'pid':os.getpid(),'stage':stage,**fields}
    with _lock:
        ROOT.mkdir(parents=True,exist_ok=True)
        with (ROOT/'events.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(item,ensure_ascii=False)+'\n')
        write_json(ROOT/'progress.json',item)
        print('STUDY_EVENT='+json.dumps(item,ensure_ascii=False),flush=True)
    return item

def copy_verified(src,dst,expected=None):
    shutil.copyfile(src,dst)
    digest=sha(dst)
    if expected and digest!=expected: raise IOError('SHA-256 mismatch: '+str(dst))
    return digest

def stage_model():
    config=json.loads((PROJECT/'project-config.json').read_text())
    assert config['deployment_authorization']['status']=='confirmed'
    variant=config['variants']['fasth3-v2-nvfp4']
    source=DRIVE/variant['drive_relative_path']/variant['revision']
    assert (source/'_COLAB_LAB_READY.json').is_file()
    receipt=json.loads((source/'_COLAB_LAB_ADALN_VIEW/ready.json').read_text())
    assert receipt['source_revision']==variant['revision'] and receipt['active_tensors_bit_identical']
    local=Path(config['runtime_root'])/'models'/variant['revision']
    if local.is_dir():
        assert (local/'_STAGED_VIEW.json').is_file()
        event('model_cache_reused',path=str(local))
        return local
    manifest=json.loads((ROOT/'model-manifest.json').read_text())
    pending=local.with_name(local.name+'.copying')
    pending.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    event('weight_staging_started')
    def copy(record):
        name=record['path']
        target=pending/name
        target.parent.mkdir(parents=True,exist_ok=True)
        replacement=receipt['files'].get(name)
        original=source/('_COLAB_LAB_ADALN_VIEW/'+name if replacement else name)
        expected=replacement['sha256'] if replacement else record.get('sha256')
        expected_bytes=replacement.get('bytes') if replacement else record['bytes']
        # Retain completed, hashed copies when a diagnosed retry reuses this VM.
        if target.is_file() and (expected_bytes is None or target.stat().st_size==expected_bytes) and (not expected or sha(target)==expected):
            digest=sha(target)
        else:
            digest=copy_verified(original,target,expected)
        if expected_bytes is not None and target.stat().st_size!=expected_bytes:
            raise IOError('Wrong staged byte count: '+name)
        event('weight_file_verified',file=name,bytes=target.stat().st_size,sha256=digest)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(copy,manifest['files']))
    write_json(pending/'_STAGED_VIEW.json',receipt)
    pending.rename(local)
    event('weight_staging_completed',seconds=time.monotonic()-started,
          bytes=sum(p.stat().st_size for p in local.rglob('*') if p.is_file()))
    return local

def freeze_assets():
    target=ROOT/'assets'
    target.mkdir(exist_ok=True)
    manifest=json.loads((ASSETS/'manifest.json').read_text())
    for name,record in manifest['files'].items():
        copy_verified(ASSETS/name,target/name,record['sha256'])
    copy_verified(ASSETS/'manifest.json',target/'manifest.json')
    event('inputs_verified',sha256=manifest['files']['first-frame.png']['sha256'])
    return target

def publish(session_dir):
    """Copy durable evidence and verify media hashes; remote flush is done by the kernel."""
    with _publish_lock:
        _publish(session_dir)

def _publish(session_dir):
    destination=RUNS/session_dir
    destination.mkdir(parents=True,exist_ok=True)
    for name in ('events.jsonl','resources.jsonl','startup-resources.jsonl','hardware.json','runtime-versions.json','worker.log','stdout.log','stderr.log','process-exit.json','state.json','progress.json'):
        path=ROOT/name
        if path.is_file(): shutil.copyfile(path,destination/name)
    for group in ('results','commands','bundles'):
        for path in (ROOT/group).glob('*'):
            if path.is_file():
                folder=destination/group
                folder.mkdir(exist_ok=True)
                target=folder/path.name
                if target.is_file() and target.stat().st_size==path.stat().st_size and path.suffix in ('.mp4','.png','.zip'):
                    continue
                copy_verified(path,target,sha(path))
    write_json(destination/'published.json',{'at':time.time(),'local_root':str(ROOT),'verified':True})

