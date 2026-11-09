"""One-second board telemetry and the inference PID tree, without psutil."""
import json
import os
from pathlib import Path
import subprocess
import threading
import time

class Sampler:
    def __init__(self,path,root_pid):
        self.path=Path(path)
        self.root_pid=root_pid
        self.stop_event=threading.Event()
        self.previous={}
        self.previous_at=time.monotonic()
        self.command=None

    def processes(self):
        rows={}
        for folder in Path('/proc').iterdir():
            if not folder.name.isdigit(): continue
            try:
                stat=(folder/'stat').read_text().rsplit(')',1)[1].split()
                rows[int(folder.name)]=(int(stat[1]),(int(stat[11])+int(stat[12]))/os.sysconf('SC_CLK_TCK'))
            except (FileNotFoundError,PermissionError,ProcessLookupError): pass
        chosen={self.root_pid}
        while True:
            grown=chosen|{pid for pid,(ppid,_) in rows.items() if ppid in chosen}
            if grown==chosen: break
            chosen=grown
        total={'rss_bytes':0,'read_bytes':0,'write_bytes':0,'cpu_seconds_delta':0.0,'pids':[]}
        for pid in chosen:
            try:
                folder=Path('/proc')/str(pid)
                rss=int((folder/'statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
                io=dict(line.split(':',1) for line in (folder/'io').read_text().splitlines())
                cpu=rows[pid][1]
                total['cpu_seconds_delta']+=max(0,cpu-self.previous.get(pid,cpu))
                self.previous[pid]=cpu
                total['rss_bytes']+=rss
                total['read_bytes']+=int(io.get('read_bytes',0))
                total['write_bytes']+=int(io.get('write_bytes',0))
                total['pids'].append(pid)
            except (FileNotFoundError,PermissionError,ProcessLookupError,KeyError): pass
        now=time.monotonic()
        total['cpu_percent']=100*total.pop('cpu_seconds_delta')/max(.001,now-self.previous_at)
        self.previous_at=now
        return total

    def start(self):
        self.thread=threading.Thread(target=self.run,daemon=True)
        self.thread.start()
        return self

    def run(self):
        fields='timestamp,index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu'
        self.command=subprocess.Popen(['nvidia-smi','--query-gpu='+fields,'--format=csv,noheader,nounits','--loop=1'],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
        with self.path.open('a',encoding='utf-8',buffering=1) as stream:
            for line in self.command.stdout:
                if self.stop_event.is_set(): break
                values=[s.strip() for s in line.split(',')]
                if len(values)!=7: continue
                def number(value):
                    try: return float(value)
                    except ValueError: return None
                row={'at':time.time(),'gpu_timestamp':values[0],'gpu_index':number(values[1]),
                     'gpu_util_percent':number(values[2]),'gpu_used_mib':number(values[3]),
                     'gpu_total_mib':number(values[4]),'power_watts':number(values[5]),
                     'temperature_c':number(values[6]),'process_tree':self.processes()}
                stream.write(json.dumps(row)+'\n')

    def stop(self):
        self.stop_event.set()
        if self.command: self.command.terminate()
        self.thread.join(timeout=5)

