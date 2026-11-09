"""Bounded CPU session for Drive authorization/preparation. No GPU requested."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

PROJECT = Path(__file__).resolve().parent
LAB = PROJECT.parent.parent
sys.path.insert(0,str(LAB))
import budget_guard as guard

POINTER = PROJECT / "cpu-session-local.json"

def start(policy=None, name_prefix="h3-cpu", pointer_path=POINTER, *, allow_preexisting_cpu=False):
    import fcntl
    lock = (guard.PRIVATE / "run.lock").open("w")
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    api = guard.client()
    before = guard.usage(api)
    existing = api.list_assignments()
    if allow_preexisting_cpu and any(item.accelerator.value != "NONE" for item in existing):
        raise RuntimeError("A preexisting GPU runtime is active; no additional runtime allocated")
    if not allow_preexisting_cpu and (existing or before["rate_cu_per_hour"] or before["active_assignments"]):
        raise RuntimeError("An existing runtime is active; refusing another allocation")
    job_id = uuid.uuid4()
    policy = policy or {"gpu":"NONE","max_seconds":1800,"max_cu":0.25,
              "max_rate_cu_per_hour":0.5,"execution_timeout_seconds":600,
              "automatic_allocation_retries":0}
    job = {"name":"budget-" + name_prefix + "-" + job_id.hex[:10],"mode":"managed_session",
           "notebook_uuid":str(job_id),"before":before,"started":time.time(),
           "deadline":time.time()+policy["max_seconds"] if policy.get("max_seconds") is not None else None,"policy":policy,
           "baseline_endpoints":[item.endpoint for item in existing]}
    path = guard.PRIVATE / (job["name"] + ".json")
    guard.write_json(path,job)
    log = path.with_suffix(".watch.log").open("w")
    watcher = subprocess.Popen([sys.executable,str(LAB/"budget_guard.py"),"_watch",str(path)],
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(150):
        if path.with_suffix(".ready").exists():
            break
        if watcher.poll() is not None:
            raise RuntimeError("CPU watchdog did not start; no VM allocated")
        time.sleep(0.1)
    else:
        watcher.terminate()
        raise RuntimeError("CPU watchdog readiness timeout; no VM allocated")
    try:
        with path.with_suffix(".worker.log").open("w") as output:
            created = subprocess.run([sys.executable,str(LAB/"budget_guard.py"),"_worker",str(path)],
                stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,timeout=90,start_new_session=True)
        if created.returncode or not guard.owned_endpoint(api,job):
            raise RuntimeError(f"{policy['gpu']} session allocation failed")
    except BaseException as exc:
        allocation_log=path.with_suffix('.worker.log')
        if isinstance(exc,subprocess.TimeoutExpired) or (allocation_log.exists() and 'ReadTimeout' in allocation_log.read_text(errors='replace')):
            guard.mark_allocation_uncertain(job)
        if guard.cleanup(api,job):
            guard.write_json(path.with_suffix(".done"),{"cleanup_verified":True})
        raise
    guard.write_json(Path(pointer_path),{"job_file":str(path),"name":job["name"],"policy":policy})
    print(json.dumps({"name":job["name"],"policy":policy,"usage":guard.usage(api)}))
    return {"job_file":str(path),"name":job["name"],"policy":policy}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["start","status","stop"])
    action=parser.parse_args().action
    if action=="start":
        return start()
    pointer=json.loads(POINTER.read_text())
    path=Path(pointer["job_file"])
    job=json.loads(path.read_text())
    api=guard.client()
    if action=="stop":
        released=guard.cleanup(api,job)
        if released:
            guard.write_json(path.with_suffix(".done"),{"cleanup_verified":True})
        print(json.dumps({"cleanup_verified":released,"usage":guard.usage(api)}))
        if not released:
            raise RuntimeError("CPU release not verified")
    else:
        print(json.dumps({"name":job["name"],"assigned":bool(guard.owned_endpoint(api,job)),"usage":guard.usage(api)}))

if __name__=="__main__":
    main()
