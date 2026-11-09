"""Execute the approved cold/matrix/compile/offload protocol, with no inference retries."""
import json
from pathlib import Path
import time
import control as c

ROOT=Path(__file__).resolve().parent
INPUT_VERSION=json.loads((ROOT/'input-version.json').read_text())['name']
CASES=json.loads((ROOT/'assets'/INPUT_VERSION/'cases.json').read_text())
CASE_MAP={case['id']:case for case in CASES}
PROGRESS=ROOT/'formal-progress.json'

def progress(phase,**fields):
    record={'at':time.time(),'phase':phase,**fields}
    c.save(PROGRESS,record)
    print('FORMAL_PROGRESS='+json.dumps(record),flush=True)

def submit(job):
    p=c.pointer()
    folder=ROOT/'scheduled'/p['session']
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/(job['id']+'.json')
    c.save(path,job)
    c.submit(path)
    return wait(job['id'])

def wait(command):
    p=c.pointer()
    files=c.files(p)
    printed=0
    while True:
        if Path(p['job_file']).with_suffix('.done').exists():
            raise RuntimeError('Watchdog released the runtime before command '+command+' completed')
        try:
            result=json.loads(c.read(files,'results/'+command+'-command.json'))
        except FileNotFoundError:
            result=None
        if result and result.get('completed'):
            progress('command-completed',session=p['session'],command=command,result=result)
            if result.get('failed') or any(not r['passed'] and not r.get('oom') for r in result.get('results',[])):
                raise RuntimeError('A non-capacity failure requires diagnosis before any further inference: '+command)
            return result
        if time.time()-printed>=30:
            try:
                state=json.loads(c.read(files,'progress.json'))
                progress('waiting',session=p['session'],command=command,stage=state.get('stage'),detail=state.get('name',state.get('file')))
            except FileNotFoundError: pass
            try:
                exit_record=json.loads(c.read(files,'process-exit.json'))
                if exit_record.get('returncode'):
                    raise RuntimeError('Inference process exited: '+str(exit_record['returncode']))
            except FileNotFoundError: pass
            printed=time.time()
        time.sleep(5)

def samples():
    p=c.pointer()
    c.fetch(media=False)
    folder=ROOT/'evidence'/p['session']/'results'
    rows=[]
    for path in folder.glob('*.json'):
        value=json.loads(path.read_text())
        if 'case' in value and 'passed' in value: rows.append(value)
    return rows

def run():
    frozen=json.loads((ROOT/'frozen-environment.json').read_text())
    assert frozen['ready'] and frozen['main_clip_technical_passed'] and frozen['main_clip_content_passed']
    assert json.loads((ROOT/'code-package.json').read_text())==frozen['code_package']
    cold=[]
    active=False
    try:
        for index in (1,2,3):
            progress('allocating-cold',index=index)
            c.allocate('cold'+str(index))
            active=True
            result=submit({'id':f'10-cold-{index}','action':'generate','cases':['480x832-5s'],
                           'repeats':1,'seed':42,'profile':'baseline','category':'cold','cold_index':index})
            if not result.get('results') or not all(r['passed'] for r in result['results']):
                raise RuntimeError('An independent cold-start sample did not pass media validation')
            c.fetch()
            cold.append({'index':index,'session':c.pointer()['session'],'result':result})
            c.save(ROOT/'formal-cold-sessions.json',cold)
            if index<3:
                c.release()
                active=False
        progress('matrix',cold_sessions=cold)
        submit({'id':'20-matrix-low-hot','action':'generate','cases':['480x832-5s'],
                'repeats':5,'seed':42,'profile':'baseline','category':'matrix_hot'})
        submit({'id':'21-matrix-other-cases','action':'generate','cases':[case['id'] for case in CASES if case['id']!='480x832-5s'],
                'repeats':6,'seed':42,'profile':'baseline','category':'matrix','stop_on_failure':False})
        rows=samples()
        groups={}
        for row in rows:
            if row['profile']=='baseline' and row['passed']:
                groups.setdefault(row['case']['id'],[]).append(row)
        stable=[key for key,values in groups.items() if len(values)>=6]
        if not stable: raise RuntimeError('No specification completed a first run and five hot runs')
        ordered=sorted(stable,key=lambda key:CASE_MAP[key]['width']*CASE_MAP[key]['height']*CASE_MAP[key]['frames'])
        representatives=[]
        for key in ('480x832-5s','768x1344-10s',ordered[-1]):
            if key in stable and key not in representatives: representatives.append(key)
        if len(representatives)<3:
            for key in sorted(ordered,key=lambda key:abs(ordered.index(key)-len(ordered)//2)):
                if key not in representatives: representatives.append(key)
                if len(representatives)>=min(3,len(stable)): break
        c.save(ROOT/'compile-representatives.json',{'cases':representatives,'selection_metric':'width * height * frames'})
        progress('vae-compile',representatives=representatives)
        submit({'id':'30-vae-compiled','action':'generate','cases':representatives,'repeats':6,
                'seed':42,'profile':'vae-compiled','category':'compile','stop_on_failure':False})
        rows=samples()
        for source_profile,target_profile,command in (
                ('baseline','cpu-offload','40-baseline-oom-offload'),
                ('vae-compiled','vae-compiled-cpu-offload','41-compiled-oom-offload')):
            failed={row['case']['id'] for row in rows if row['profile']==source_profile and row.get('oom')}
            cases=[case['id'] for case in CASES if case['id'] in failed]
            if cases:
                progress('cpu-offload',source_profile=source_profile,cases=cases)
                submit({'id':command,'action':'generate','cases':cases,'repeats':6,'seed':42,
                        'profile':target_profile,'category':'offload','stop_on_failure':False})
        progress('publishing-all-evidence')
        submit({'id':'90-package-evidence','action':'package'})
        c.fetch()
        progress('releasing-final-gpu')
        c.release()
        active=False
        progress('gpu-tests-finished',cold_sessions=cold)
    finally:
        if active: c.release()

if __name__=='__main__': run()
