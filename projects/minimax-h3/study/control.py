"""Host-side study control; never stores OAuth or runtime proxy tokens on Drive."""
import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import hashlib
import zipfile
from datetime import datetime,timedelta,timezone

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
LAB=PROJECT.parent.parent
sys.path[:0]=[str(LAB),str(PROJECT)]
import budget_guard as guard
from cpu_prepare_session import start
from colab_cli.contents import ContentsClient
from colab_cli.state import StateStore

POINTER=HERE/'current-session.json'
REMOTE='/content/colab-lab/whale-study'
POLICY=dict(gpu='G4',max_seconds=900,max_cu=2.5,max_rate_cu_per_hour=12.,min_rate_cu_per_hour=9.,
            balance_floor_cu=10.,stall_seconds=1800,execution_timeout_seconds=None,automatic_allocation_retries=0,
            monitor_failure_grace_seconds=45,monitor_transport_retries=2,cleanup_reserve_seconds=90)

def save(path,value): guard.write_json(Path(path),value)

def pointer(): return json.loads(POINTER.read_text())

def files(p):
    def current_session():
        store=StateStore(str(guard.STATE))
        session=store.get(p['name'])
        if session is None: raise RuntimeError('Owned session is absent from local state')
        if not session.token_expires_at or session.token_expires_at-datetime.now(timezone.utc)<=timedelta(minutes=5):
            assignment=next((a for a in guard.client().list_assignments() if a.endpoint==session.endpoint),None)
            if assignment is None: raise RuntimeError('Owned assignment is no longer active')
            info=assignment.runtime_proxy_info
            session.token=info.token
            session.url=info.url
            session.token_expires_at=info.expires_at()
            store.add(session)
        return session
    session=current_session()
    client=ContentsClient(session)
    # Pinned CLI 0.7.5 omits requests timeouts. Add a bounded transport here
    # without changing installed code or leaking the private proxy token.
    import requests
    from urllib.parse import quote
    def request(method,path,params=None,json_data=None):
        # Long studies outlive the one-hour runtime proxy credential. Refresh
        # the connection only; never allocate or restart a VM here.
        current=current_session()
        client.base_url=current.url.rstrip('/')
        client.token=current.token
        for attempt in range(3):
            try:
                response=requests.request(method,client.base_url+'/api/contents/'+quote(path.strip('/'),safe='/'),
                    params={'authuser':'0','colab-runtime-proxy-token':client.token,**(params or {})},
                    json=json_data,timeout=(10,25))
                break
            except requests.exceptions.RequestException as exc:
                # Read transport retries cannot enqueue a second inference.
                if method!='GET' or attempt==2:
                    raise RuntimeError('Runtime file transport '+type(exc).__name__) from None
                print('Read transport retry: '+type(exc).__name__,file=sys.stderr,flush=True)
                time.sleep((2,5)[attempt])
        if response.status_code==404: raise FileNotFoundError(path)
        if not response.ok: raise RuntimeError('Runtime file transport HTTP '+str(response.status_code))
        return None if method=='DELETE' else response.json()
    client._request=request
    return client

def read(client,name):
    data=client._request('GET',REMOTE+'/'+name,params={'content':'1'})
    raw=data['content']
    if data.get('format')=='base64': return base64.b64decode(raw).decode('utf-8')
    return raw

def cli(p,*args,timeout=60,**kwargs):
    return subprocess.run([str(guard.CLI),'--auth=oauth2','--config',str(guard.STATE),*args,'-s',p['name']],
                          timeout=timeout,**kwargs)

def event(p,stage,**fields):
    target=HERE/'evidence'/p['session']
    target.mkdir(parents=True,exist_ok=True)
    with (target/'host-events.jsonl').open('a') as out:
        out.write(json.dumps({'at':time.time(),'stage':stage,**fields})+'\n')
    job_path=Path(p['job_file'])
    save(job_path.with_suffix('.progress.json'),{'at':time.time(),'stage':stage})

def mirror():
    p=pointer()
    client=files(p)
    job_path=Path(p['job_file'])
    last=None
    last_io=0
    last_io_check=0
    while not job_path.with_suffix('.done').exists():
        try:
            item=json.loads(read(client,'progress.json'))
            if item['at']!=last:
                save(job_path.with_suffix('.progress.json'),item)
                last=item['at']
                with (HERE/'evidence'/p['session']/'remote-events.jsonl').open('a') as out:
                    out.write(json.dumps(item)+'\n')
            if item.get('stage') in ('weight_staging_started','weight_file_verified') and time.time()-last_io_check>30:
                last_io_check=time.time()
                latest=json.loads(read(client,'resources.jsonl').splitlines()[-1])
                written=latest['process_tree']['write_bytes']
                if written-last_io>=64*1024**2:
                    save(job_path.with_suffix('.progress.json'),{'at':latest['at'],
                         'stage':'weight_copy_bytes_advancing','process_tree_write_bytes':written})
                    last_io=written
        except Exception:
            # The independent billing guard expires the substantive-progress
            # timestamp after 30 minutes even if this mirror loses transport.
            pass
        time.sleep(8)

def allocate(label):
    if os.environ.get('COLAB_MEDIA_ALLOW_GPU')!='1':
        raise RuntimeError('GPU allocation disabled in publication copy; read docs/deployment.md first')
    if label.startswith('cold'):
        frozen=json.loads((HERE/'frozen-environment.json').read_text())
        if not frozen.get('ready'): raise RuntimeError('A verified main clip and frozen environment are required')
    before=guard.usage(guard.client())
    if before['balance_cu']<=POLICY['balance_floor_cu']+1:
        raise RuntimeError('Protected balance: no GPU allocated')
    p=start(POLICY,'h3-study',POINTER)
    p['session']=time.strftime('%Y%m%d-%H%M%S',time.gmtime())+'-'+label
    save(POINTER,p)
    target=HERE/'evidence'/p['session']
    target.mkdir(parents=True,exist_ok=True)
    save(target/'allocation.json',p)
    event(p,'instance_assigned')
    print('STUDY_SESSION='+p['session'],flush=True)
    print('Drive mount starting; any current binding is logged in CLI history.',flush=True)
    try:
        options={'session':p['session'],'purpose':label,'policy':POLICY,'code_version':'whale-study-v2',
                 'code_package':json.loads((HERE/'code-package.json').read_text()),
                 'assets_version':json.loads((HERE/'input-version.json').read_text())['name']}
        init=target/'init_remote.py'
        sampler=(HERE/'resource_sampler.py').read_text()
        loop='''import sys,time
from pathlib import Path
from resource_sampler import Sampler
root=Path('/content/colab-lab/whale-study')
sampler=Sampler(root/'startup-resources.jsonl',int(sys.argv[1])).start()
while not (root/'stop-startup-sampler').exists(): time.sleep(1)
sampler.stop()
(root/'startup-sampler-done.json').write_text('{}')
'''
        init.write_text('from pathlib import Path\nimport subprocess,sys,os,json\np=Path('+repr(REMOTE)+')\np.mkdir(parents=True,exist_ok=True)\n'
            +'(p/"commands").mkdir(exist_ok=True)\n(p/"results").mkdir(exist_ok=True)\n'
            +'(p/"session-options.json").write_text('+repr(json.dumps(options))+')\n'
            +'(p/"resource_sampler.py").write_text('+repr(sampler)+')\n'
            +'(p/"startup_sampler.py").write_text('+repr(loop)+')\n'
            +'log=(p/"startup-sampler.log").open("w")\n'
            +'child=subprocess.Popen([sys.executable,str(p/"startup_sampler.py"),str(os.getpid())],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)\n'
            +'print("STARTUP_SAMPLER_STARTED",flush=True)\n',encoding='utf-8')
        setup=cli(p,'exec','-f',str(init),'--timeout','60',timeout=80,capture_output=True,text=True)
        if setup.returncode or 'STARTUP_SAMPLER_STARTED' not in setup.stdout:
            raise RuntimeError('Startup sampler initialization failed')
        event(p,'startup_resource_sampler_started')
        event(p,'drive_mount_started')
        with (target/'mount.log').open('w') as log:
            result=cli(p,'drivemount','/content/drive',timeout=1810,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            text=(target/'mount.log').read_text()
            if 'Credentials propagated' in text and 'mount failed' in text:
                with (target/'mount.log').open('a') as log:
                    result=cli(p,'drivemount','/content/drive',timeout=60,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode: raise RuntimeError('Drive mount failed; preserved mount log')
        event(p,'drive_mount_completed')
        with (target/'mirror.log').open('a') as log:
            mirror_process=subprocess.Popen([sys.executable,str(__file__),'_mirror'],stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        p['mirror_pid']=mirror_process.pid
        save(POINTER,p)
        event(p,'bootstrap_started')
        with (target/'bootstrap.log').open('w') as log:
            result=cli(p,'exec','-f',str(HERE/'bootstrap_remote.py'),'--timeout','1800',timeout=1820,
                       stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONUNBUFFERED='1'))
        text=(target/'bootstrap.log').read_text()
        if result.returncode or 'BOOTSTRAP_RESULT=' not in text:
            raise RuntimeError('Bootstrap did not produce a worker launch receipt')
        event(p,'bootstrap_completed')
        print(json.dumps({'session':p['session'],'launched':True,'usage':guard.usage(guard.client())}),flush=True)
    except BaseException:
        release()
        raise

def submit(path):
    p=pointer()
    data=json.loads(Path(path).read_text())
    if data['action'] not in ('generate','stop','package'): raise ValueError('Unsupported job')
    if not data['id'].replace('-','').replace('_','').isalnum(): raise ValueError('Unsafe command id')
    target=HERE/'evidence'/p['session']/'commands'
    target.mkdir(exist_ok=True)
    destination=target/(data['id']+'.json')
    if destination.exists():
        if json.loads(destination.read_text())!=data: raise RuntimeError('Command id already has different content')
        try:
            read(files(p),'commands/'+destination.name)
        except FileNotFoundError:
            pass  # A prior file-transport failure did not enqueue inference.
        else: raise RuntimeError('Command already submitted; inspect its original evidence')
    save(destination,data)
    files(p).upload(str(destination),REMOTE+'/commands/'+destination.name)
    event(p,'command_submitted',command=data['id'])
    print(json.dumps({'submitted':data['id']}))

def status():
    p=pointer()
    client=files(p)
    result={'session':p['session'],'usage':guard.usage(guard.client())}
    for name in ('state.json','progress.json'):
        try: result[name]=json.loads(read(client,name))
        except FileNotFoundError: pass
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if result.get('state.json',{}).get('status') in ('fatal','command_failed'):
        print(read(client,'worker.log')[-10000:])

def fetch(media=True):
    p=pointer()
    client=files(p)
    target=HERE/'evidence'/p['session']
    saved=[]
    has_bundles=False
    if media:
        try:
            manifests=[x for x in client.list_dir(REMOTE+'/bundles').get('content',[]) if x['name'].endswith('-manifest.json')]
        except FileNotFoundError: manifests=[]
        for item in manifests:
            manifest=json.loads(read(client,'bundles/'+item['name']))
            folder=target/'bundles'
            folder.mkdir(exist_ok=True)
            save(folder/item['name'],manifest)
            for chunk in manifest['chunks']:
                path=folder/chunk['name']
                def digest():
                    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
                if not path.is_file() or digest()!=chunk['sha256']:
                    client.download(REMOTE+'/bundles/'+chunk['name'],str(path))
                if digest()!=chunk['sha256']: raise IOError('Downloaded evidence archive checksum mismatch')
                with zipfile.ZipFile(path) as archive:
                    if any('..' in Path(n).parts or Path(n).is_absolute() for n in archive.namelist()):
                        raise RuntimeError('Unsafe evidence archive path')
                    archive.extractall(target)
            for name,record in manifest['files'].items():
                path=target/name
                with path.open('rb') as stream:
                    if hashlib.file_digest(stream,'sha256').hexdigest()!=record['sha256']:
                        raise IOError('Extracted evidence checksum mismatch: '+name)
            has_bundles=True
    for name in ('state.json','progress.json','events.jsonl','resources.jsonl','startup-resources.jsonl','hardware.json','runtime-versions.json','worker.log','stdout.log','stderr.log','process-exit.json'):
        try: client.download(REMOTE+'/'+name,str(target/name))
        except FileNotFoundError: pass
    for item in client.list_dir(REMOTE+'/results').get('content',[]):
        if item['type']!='file': continue
        folder=target/'results'
        folder.mkdir(exist_ok=True)
        path=folder/item['name']
        if path.suffix in ('.mp4','.png') and (not media or has_bundles): continue
        if path.is_file() and path.stat().st_size==item.get('size'): continue
        client.download(REMOTE+'/results/'+item['name'],str(path))
        saved.append({'name':path.name,'bytes':path.stat().st_size})
    print(json.dumps({'downloaded':saved},ensure_ascii=False))

def release():
    p=pointer()
    job_path=Path(p['job_file'])
    job=json.loads(job_path.read_text())
    api=guard.client()
    target=HERE/'evidence'/p['session']
    released=False
    event(p,'release_started')
    try:
        if guard.owned_endpoint(api,job):
            client=files(p)
            inference_pid=int(read(client,'inference.pid'))
            stop=target/'zz-stop.json'
            save(stop,{'id':'zz-stop','action':'stop'})
            try:
                client.upload(str(stop),REMOTE+'/commands/zz-stop.json')
                for _ in range(30):
                    try:
                        exit_receipt=json.loads(read(client,'process-exit.json'))
                        if exit_receipt.get('inference_pid')==inference_pid and 'returncode' in exit_receipt: break
                    except FileNotFoundError: pass
                    time.sleep(1)
                fetch()
            except Exception as exc: print('Evidence fetch: '+type(exc).__name__)
            marker=target/'stop-startup-sampler'
            marker.write_text('stop')
            client.upload(str(marker),REMOTE+'/stop-startup-sampler')
            for _ in range(5):
                try:
                    read(client,'startup-sampler-done.json')
                    break
                except FileNotFoundError: time.sleep(1)
            try: client.download(REMOTE+'/startup-resources.jsonl',str(target/'startup-resources.jsonl'))
            except FileNotFoundError: pass
            flush=target/'flush_remote.py'
            receipt_name='flush-receipt-'+uuid.uuid4().hex+'.json'
            flush.write_text('from pathlib import Path\nimport json,traceback\nresult={"flushed":False}\ntry:\n'
                +' from google.colab import drive\n drive.flush_and_unmount(timeout_ms=60000)\n result["flushed"]=True\n'
                +'except BaseException as exc:\n result["error"]=str(exc)\nfinally:\n Path('+repr(REMOTE+'/'+receipt_name)+').write_text(json.dumps(result))\n',encoding='utf-8')
            client.upload(str(flush),REMOTE+'/flush_remote.py')
            event(p,'drive_flush_started')
            # CLI 0.7.5 exec can hang waiting for kernel messages after CUDA JIT.
            # Its purpose-built console runs the same official flush function.
            private_log=guard.PRIVATE/(p['name']+'.flush-console.log')
            with private_log.open('w') as log:
                console=subprocess.Popen([str(guard.CLI),'--auth=oauth2','--config',str(guard.STATE),
                     'console','-s',p['name']],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,text=True)
                try:
                    console.stdin.write('python '+REMOTE+'/flush_remote.py\n')
                    console.stdin.flush()
                    deadline=time.time()+90
                    while time.time()<deadline:
                        try:
                            receipt=json.loads(read(client,receipt_name))
                            save(target/'flush-result.json',receipt)
                            if not receipt.get('flushed'): raise RuntimeError('Official Drive flush failed: '+receipt.get('error','unknown'))
                            event(p,'drive_flush_completed')
                            break
                        except FileNotFoundError: time.sleep(2)
                    else: raise RuntimeError('Official Drive flush receipt was not observed')
                finally:
                    console.stdin.close()
                    try: console.wait(timeout=5)
                    except subprocess.TimeoutExpired: console.terminate()
    finally:
        save(job_path.with_suffix('.release-requested.json'),{'at':time.time(),'persistence_attempt_finished':True})
        released=guard.cleanup(api,job)
        event(p,'instance_release_confirmed',verified=released)
        if released and not job_path.with_suffix('.done').exists(): save(job_path.with_suffix('.done'),{'cleanup_verified':True})
        report={'cleanup_verified':released,'session':p['session'],'before':job['before'],
                'started_at':job['started'],'released_at':job_path.with_suffix('.done').stat().st_mtime if released else None,
                'controller_cleanup_checked_at':time.time(),'usage':guard.usage(api)}
        for _ in range(15):
            audit=job_path.with_suffix('.release-audit.json')
            if audit.exists():
                report['independent_release_audit']=json.loads(audit.read_text())
                break
            time.sleep(1)
        for source in (job_path.with_suffix('.billing.jsonl'),job_path.with_suffix('.watch.log')):
            if source.is_file(): __import__('shutil').copyfile(source,target/source.name)
        save(target/'session-result.json',report)
        print(json.dumps(report,indent=2))
        if not released: raise RuntimeError('Independent watchdog is still reconciling release')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['start','submit','status','fetch','release','_mirror'])
    parser.add_argument('argument',nargs='?')
    args=parser.parse_args()
    if args.action=='start': allocate(args.argument or 'research')
    elif args.action=='submit': submit(args.argument)
    elif args.action=='fetch': fetch(media=args.argument!='metrics')
    else: globals()[{'_mirror':'mirror'}.get(args.action,args.action)]()

if __name__=='__main__': main()
