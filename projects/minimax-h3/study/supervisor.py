"""Capture both streams and the real inference-process exit code."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

root=Path(__file__).resolve().parent
options=json.loads((root/'session-options.json').read_text())
destination=Path('/content/drive/MyDrive/Colab-Lab/runs/minimax-h3-study/example-study')/options['session']
destination.mkdir(parents=True,exist_ok=True)
lock=threading.Lock()
started=time.time()
child=subprocess.Popen([sys.executable,str(root/'worker.py')],stdin=subprocess.DEVNULL,
    stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',errors='replace',bufsize=1)
(root/'inference.pid').write_text(str(child.pid))
with (root/'worker.log').open('a',encoding='utf-8',buffering=1) as combined:
    def pump(source,name):
        with (root/name).open('a',encoding='utf-8',buffering=1) as stream:
            for line in source:
                stream.write(line)
                with lock: combined.write(line)
        source.close()
    threads=[threading.Thread(target=pump,args=(child.stdout,'stdout.log')),
             threading.Thread(target=pump,args=(child.stderr,'stderr.log'))]
    for thread in threads: thread.start()
    code=child.wait()
    for thread in threads: thread.join()
receipt={'inference_pid':child.pid,'supervisor_pid':os.getpid(),'returncode':code,'started_at':started,
         'finished_at':time.time(),'stdout':'stdout.log','stderr':'stderr.log','combined':'worker.log'}
(root/'process-exit.json').write_text(json.dumps(receipt,indent=2))
for name in ('process-exit.json','stdout.log','stderr.log','worker.log'):
    shutil.copyfile(root/name,destination/name)
sys.exit(code)
