"""Restore a pinned runtime and launch a logged, command-driven I2V worker."""
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

root=Path('/content/colab-lab/whale-study')
root.mkdir(parents=True,exist_ok=True)
options=json.loads((root/'session-options.json').read_text())
code=Path('/content/drive/MyDrive/Colab-Lab/notebooks/minimax-h3/whale-study-v2')
package=json.loads((root/'code-package.json').read_text()) if (root/'code-package.json').is_file() else options['code_package']
bundle=code/package['name']
assert hashlib.sha256(bundle.read_bytes()).hexdigest()==package['sha256']
with zipfile.ZipFile(bundle) as archive:
    assert all(Path(name).name==name for name in archive.namelist())
    archive.extractall(root)
manifest=json.loads((root/'code-manifest.json').read_text())
for name,digest in manifest['files'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
sys.path.insert(0,str(root))
from study_runtime import event,write_json
import torch
name=torch.cuda.get_device_name(0)
assert 'RTX PRO 6000' in name.upper() and torch.cuda.get_device_properties(0).total_memory>=90*1024**3
hardware={'gpu':name,'memory_bytes':torch.cuda.get_device_properties(0).total_memory,
          'os':platform.platform(),'base_python':sys.version,'base_torch':torch.__version__,
          'base_cuda':torch.version.cuda,'cpu_count':os.cpu_count(),
          'cpu_affinity_count':len(os.sched_getaffinity(0)),
          'meminfo':Path('/proc/meminfo').read_text(),
          'local_disk_bytes':dict(zip(('total','used','free'),shutil.disk_usage('/content'))),
          'gpu_driver_and_power':subprocess.check_output(['nvidia-smi',
              '--query-gpu=name,driver_version,memory.total,power.limit',
              '--format=csv,noheader,nounits'],text=True).strip()}
for cgroup in ('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/cpu.max'):
    if Path(cgroup).is_file(): hardware[cgroup]=Path(cgroup).read_text().strip()
write_json(root/'hardware.json',hardware)
event('instance_hardware_verified',**hardware)
assert os.path.ismount('/content/drive')
python=Path('/content/colab-lab/h3-environment/.venv/bin/python')
if not python.is_file():
    project=code.parent
    record=json.loads((project/'cache/environment-ready.json').read_text())
    expected=record['sha256']
    assert record['sha256']==expected
    event('environment_copy_started')
    started=time.monotonic()
    local=root.parent/'verified-h3-environment.tar.gz'
    digest=hashlib.sha256()
    with Path(record['archive']).open('rb',buffering=16*1024**2) as source,local.open('wb') as target:
        while data:=source.read(8*1024**2):
            target.write(data)
            digest.update(data)
    assert digest.hexdigest()==expected
    event('environment_archive_verified',seconds=time.monotonic()-started,bytes=local.stat().st_size)
    unpack=time.monotonic()
    subprocess.run(['tar','-xzf',str(local),'-C',str(root.parent),'--no-same-owner','--no-same-permissions'],check=True)
    event('environment_restored',seconds=time.monotonic()-unpack,total_seconds=time.monotonic()-started)
overlay_record=json.loads((code.parent/'cache/fp4-kernel-overlay-ready.json').read_text())
assert overlay_record['source_revision']=='ce970eb715215c79a169658d7aa6c8a96e64891e'
overlay_archive=root/'fp4-kernel-overlay.tar.gz'
shutil.copyfile(overlay_record['archive'],overlay_archive)
assert hashlib.sha256(overlay_archive.read_bytes()).hexdigest()==overlay_record['archive_sha256']
subprocess.run(['tar','-xzf',str(overlay_archive),'-C',str(root.parent),'--no-same-owner','--no-same-permissions'],check=True)
event('pinned_fp4_kernel_overlay_restored',bytes=overlay_archive.stat().st_size)
if (root/'worker.pid').exists():
    pid=int((root/'worker.pid').read_text())
    if Path('/proc',str(pid)).exists(): raise RuntimeError('Existing study worker must be inspected first')
env=dict(os.environ,PYTHONUNBUFFERED='1',WANDB_MODE='disabled',HF_HUB_DISABLE_TELEMETRY='1',
         PATH=str(python.parent)+os.pathsep+os.environ.get('PATH',''))
assert (python.parent/'ninja').is_file(),'Cached environment needs its ninja build tool'
with (root/'supervisor.log').open('a',buffering=1) as log:
    child=subprocess.Popen([str(python),str(root/'supervisor.py')],stdout=log,stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL,start_new_session=True,env=env,cwd=root)
(root/'worker.pid').write_text(str(child.pid))
print('BOOTSTRAP_RESULT='+json.dumps({'pid':child.pid,'root':str(root),'session':options['session']}),flush=True)

