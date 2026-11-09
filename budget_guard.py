"""Budgeted T4 verification. Pinned to google-colab-cli 0.7.5.

The CU threshold is an early-stop target, not a provider billing hard limit.
An independent watchdog survives closing the invoking terminal while WSL stays up.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import types
import uuid
import re

os.umask(0o077)
os.environ["COLAB_CLI_REQUEST_TIMEOUT"] = "12"
os.environ["COLAB_CLI_ASSIGN_RETRIES"] = "0"
ROOT = Path(__file__).resolve().parent
PRIVATE = Path.home() / ".config/colab-lab"
PRIVATE.mkdir(parents=True, exist_ok=True)
STATE = PRIVATE / "sessions.json"
CLI = Path(sys.executable).with_name("colab")
POLICY = {"gpu": "T4", "max_seconds": 180, "max_cu": 0.25,
          "max_rate_cu_per_hour": 3.0, "execution_timeout_seconds": 60,
          "automatic_allocation_retries": 0}

def write_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp.replace(path)

def client():
    token = Path.home() / ".config/colab-cli/token.json"
    if not token.is_file():
        raise RuntimeError("Sign in first: .\\lab.ps1 login (no GPU is allocated by login)")
    from colab_cli.auth import get_credentials, AuthProvider
    from colab_cli.client import Client, Prod
    return Client(Prod(), get_credentials(provider=AuthProvider.OAUTH2))

def usage(api):
    info = api.get_consumption_user_info()
    result = {"balance_cu": info.paid_compute_units_balance,
              "rate_cu_per_hour": info.consumption_rate_hourly,
              "active_assignments": info.assignments_count}
    if not all(math.isfinite(result[k]) and result[k] >= 0 for k in ("balance_cu", "rate_cu_per_hour")):
        raise RuntimeError("Invalid billing information; refusing to proceed")
    return result

def owned_endpoint(api, job):
    # Read-only lookup with the job's fixed notebook UUID also recovers an
    # allocation whose HTTP response was lost before CLI state could be saved.
    from colab_cli.client import Variant, Accelerator, Assignment
    gpu = job.get("policy", POLICY)["gpu"]
    variant = Variant.DEFAULT if gpu == "NONE" else Variant.GPU
    result = api._get_assignment(uuid.UUID(job["notebook_uuid"]), variant, Accelerator(gpu))
    return result.endpoint if isinstance(result, Assignment) else None

def mark_allocation_uncertain(job):
    # A timed-out POST can finish later. Keep the independent watcher alive
    # for this original UUID, without ever sending another allocation POST.
    write_json(PRIVATE / (job["name"] + ".allocation-uncertain.json"),
               {"until": min(job.get("deadline") or float('inf'), time.time()+300)})

def cleanup(api, job):
    pending = PRIVATE / (job.get("name", "unnamed") + ".allocation-uncertain.json")
    uncertain_until=json.loads(pending.read_text())["until"] if pending.exists() else 0
    for attempt in range(3):
        try:
            endpoint = owned_endpoint(api, job)
            if endpoint:
                api.unassign(endpoint)
            if not owned_endpoint(api, job):
                if not endpoint and time.time()<uncertain_until:
                    return False
                if endpoint and pending.exists():
                    pending.unlink()
                return True
        except Exception as exc:
            print(f"Cleanup attempt {attempt + 1}: {type(exc).__name__}", flush=True)
        time.sleep(1 + attempt)
    return False

def stop_reason(job, current, assignments, now):
    policy = job.get("policy", POLICY)
    if job.get("deadline") is not None and now >= job["deadline"]:
        return "wall-clock deadline"
    rate = current["rate_cu_per_hour"]
    if rate > policy["max_rate_cu_per_hour"]:
        return "hourly rate exceeds selected hardware policy"
    spent = max(0.0, job["before"]["balance_cu"] - current["balance_cu"],
                max(0, now - job["started"]) * rate / 3600)
    # Reserve 20 seconds of current-rate billing for monitoring and teardown.
    reserve = rate * policy.get('cleanup_reserve_seconds',20) / 3600
    if policy.get("max_cu") is not None and spent + reserve >= policy["max_cu"]:
        return "CU threshold including cleanup reserve"
    if policy.get("balance_floor_cu") is not None:
        estimated_balance=min(current["balance_cu"],job["before"]["balance_cu"]-spent)
        if estimated_balance-reserve <= policy["balance_floor_cu"]:
            return "protected remaining CU balance"
    if policy.get("stall_seconds"):
        last_progress=job.get("last_progress",job["started"])
        if now-last_progress>=policy["stall_seconds"]:
            return "no substantive progress within stall limit"
    if policy.get("min_rate_cu_per_hour") and now-job["started"]>60 and rate<policy["min_rate_cu_per_hour"]:
        return "hourly rate below expected G4 range; verify allocation"
    baseline_endpoints = set(job.get("baseline_endpoints", []))
    relevant = [a for a in assignments if getattr(a,"endpoint",None) not in baseline_endpoints]
    if any(str(a.accelerator.value) != policy["gpu"] for a in relevant):
        return "unexpected accelerator"
    if len(relevant) > 1:
        return "unexpected concurrent assignment; stop our job only"
    return None

def monitor_retry_delay(job,failures,first_failed_at,now):
    policy=job.get('policy',POLICY)
    grace=policy.get('monitor_failure_grace_seconds',0)
    if not grace or failures>policy.get('monitor_transport_retries',2) or now-first_failed_at>=grace:
        return None
    return min(15,grace-(now-first_failed_at))

def watchdog(job_path):
    job = json.loads(job_path.read_text())
    done = job_path.with_suffix(".done")
    api = client()
    write_json(job_path.with_suffix(".ready"), {"pid": os.getpid()})
    last_reason = None
    last_usage=None
    last_assignments=[]
    monitor_failure_started=None
    monitor_failures=0
    while True:
        if done.exists():
            # Independently verify the controller's release claim, including
            # every exception path; do not merely trust its .done marker.
            if cleanup(api,job):
                write_json(job_path.with_suffix('.release-audit.json'),
                           {"cleanup_verified":True,"verified_at":time.time(),"by":"independent watchdog"})
                return
            time.sleep(3)
            continue
        if job_path.with_suffix('.allocation-uncertain.json').exists():
            if cleanup(api,job):
                write_json(done,{"cleanup_verified":True,"reason":"uncertain allocation reconciled"})
                write_json(job_path.with_suffix('.release-audit.json'),
                           {"cleanup_verified":True,"verified_at":time.time(),"by":"independent watchdog"})
                return
            time.sleep(3)
            continue
        if not job_path.with_suffix(".pid").exists() and (job.get("deadline") is None or time.time() < job["deadline"]):
            time.sleep(0.1)
            continue
        try:
            if job_path.with_suffix('.release-requested.json').exists():
                last_reason=last_reason or 'controller requested release after persistence attempt'
            current = usage(api)
            last_usage=current
            assignments = api.list_assignments()
            last_assignments=assignments
            monitor_failure_started=None
            monitor_failures=0
            progress=job_path.with_suffix('.progress.json')
            if progress.exists():
                job['last_progress']=json.loads(progress.read_text()).get('at',job['started'])
            if job['policy'].get('balance_floor_cu') is not None:
                with job_path.with_suffix('.billing.jsonl').open('a') as stream:
                    stream.write(json.dumps({'at':time.time(),**current})+'\n')
            reason = last_reason or stop_reason(job, current, assignments, time.time())
        except Exception as exc:
            now=time.time()
            monitor_failures+=1
            if monitor_failure_started is None: monitor_failure_started=now
            delay=monitor_retry_delay(job,monitor_failures,monitor_failure_started,now)
            if delay is not None and last_reason is None and isinstance(exc,(OSError,TimeoutError)):
                cached=dict(last_usage or job['before'])
                cached['rate_cu_per_hour']=cached.get('rate_cu_per_hour') or job['policy']['max_rate_cu_per_hour']
                protected=stop_reason(job,cached,last_assignments,now)
                if protected is None:
                    match=re.search(r'\[SSL:\s*([A-Z_]+)\]',str(exc))
                    write_json(job_path.with_suffix('.monitor-warning.json'),
                        {'at':now,'first_failed_at':monitor_failure_started,'attempt':monitor_failures,
                         'exception':type(exc).__name__,'ssl_reason':match.group(1) if match else None,
                         'retry_delay_seconds':delay,'max_transport_retries':job['policy'].get('monitor_transport_retries',2)})
                    time.sleep(delay)
                    continue
                last_reason=protected
            reason = last_reason or f"billing monitor unavailable: {type(exc).__name__}"
        if reason:
            last_reason = reason
            write_json(job_path.with_suffix(".stopped"), {"reason": reason})
            # Terminate the allocating/executing CLI before cleanup, so it cannot
            # create another assignment after the cleanup check.
            if job_path.with_suffix(".pid").exists():
                worker_pid = int(job_path.with_suffix(".pid").read_text())
                try:
                    arguments = (Path("/proc") / str(worker_pid) / "cmdline").read_bytes().split(b"\0")
                    if b"_worker" in arguments and str(job_path).encode() in arguments:
                        os.killpg(worker_pid, signal.SIGTERM)
                except (ProcessLookupError,FileNotFoundError):
                    pass
            if cleanup(api, job):
                write_json(done, {"cleanup_verified": True, "reason": last_reason})
                write_json(job_path.with_suffix('.release-audit.json'),
                           {"cleanup_verified":True,"verified_at":time.time(),"by":"independent watchdog"})
                return
            # Keep retrying cleanup; never report a successful release on error.
            time.sleep(5)
        else:
            time.sleep(3)

def worker(job_path):
    if os.environ.get('COLAB_MEDIA_ALLOW_GPU')!='1':
        raise RuntimeError('GPU allocation disabled in publication copy')
    job = json.loads(job_path.read_text())
    job_path.with_suffix(".pid").write_text(str(os.getpid()))
    # Bind only this command module's UUID provider; installed CLI files and the
    # system-wide UUID module remain unchanged. This lets the watchdog recover
    # exactly our allocation after a lost response, without touching other jobs.
    import colab_cli.commands.run as run_module
    run_module.uuid = types.SimpleNamespace(uuid4=lambda: uuid.UUID(job["notebook_uuid"]))
    from colab_cli.cli import main
    if job.get("mode") == "managed_session":
        import colab_cli.commands.session as session_module
        session_module.uuid = types.SimpleNamespace(uuid4=lambda: uuid.UUID(job["notebook_uuid"]))
        policy = job["policy"]
        os.environ["COLAB_CLI_REQUEST_TIMEOUT"] = "60"
        gpu_args = [] if policy["gpu"] == "NONE" else ["--gpu", policy["gpu"]]
        sys.argv = [str(CLI), "--auth=oauth2", "--config", str(STATE), "new", "-s", job["name"], *gpu_args]
        main()
        return
    policy = job.get("policy", POLICY)
    gpu_args = [] if policy["gpu"] == "NONE" else ["--gpu", policy["gpu"]]
    sys.argv = [str(CLI), "--auth=oauth2", "--config", str(STATE), "run",
                "-s", job["name"], *gpu_args, "--timeout", str(policy["execution_timeout_seconds"]),
                job.get("script", str(ROOT / "gpu_smoke.py"))]
    main()

def run_smoke():
    return run_job(ROOT / "gpu_smoke.py", POLICY, ROOT / "verification-result.json")

def run_job(script, policy, result_path, *, name_prefix="smoke"):
    import fcntl
    script, result_path = Path(script).resolve(), Path(result_path).resolve()
    if policy["gpu"] not in ("NONE", "T4", "G4", "L4", "A100", "H100"):
        raise ValueError("Unsupported accelerator")
    for key in ("max_seconds", "max_cu", "max_rate_cu_per_hour", "execution_timeout_seconds"):
        if not math.isfinite(policy[key]) or policy[key] <= 0:
            raise ValueError("Budget/time limits must be positive and finite")
    if not name_prefix.replace("-", "").isalnum():
        raise ValueError("Use an alphanumeric job prefix")
    lock = (PRIVATE / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    compile(script.read_text(encoding="utf-8"), script.name, "exec")
    api = client()
    before = usage(api)
    if before["active_assignments"] or before["rate_cu_per_hour"] or api.list_assignments():
        raise RuntimeError("Existing sessions/running charges detected; no new VM was allocated")
    if before["balance_cu"] < 1:
        raise RuntimeError("Less than 1 CU remaining; no VM was allocated")
    job_id = uuid.uuid4()
    job = {"name": "budget-" + name_prefix + "-" + job_id.hex[:10], "notebook_uuid": str(job_id),
           "before": before, "started": time.time(), "deadline": time.time() + policy["max_seconds"],
           "policy": policy, "script": str(script)}
    job_path = PRIVATE / (job["name"] + ".json")
    write_json(job_path, job)
    watch_log = job_path.with_suffix(".watch.log").open("w")
    watcher = subprocess.Popen([sys.executable, __file__, "_watch", str(job_path)],
                               stdin=subprocess.DEVNULL, stdout=watch_log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    ready = job_path.with_suffix(".ready")
    for _ in range(150):
        if ready.exists():
            break
        if watcher.poll() is not None:
            raise RuntimeError("Watchdog failed; no VM was allocated")
        time.sleep(0.1)
    else:
        watcher.terminate()
        raise RuntimeError("Watchdog not ready; no VM was allocated")
    print(json.dumps({"before": before, "policy": policy, "watchdog_pid": watcher.pid}), flush=True)
    worker_log_path = job_path.with_suffix(".worker.log")
    process = None
    exit_code = -1
    try:
        with worker_log_path.open("w") as log:
            process = subprocess.Popen([sys.executable, __file__, "_worker", str(job_path)],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            exit_code = process.wait(timeout=policy["max_seconds"] + 5)
    finally:
        if process and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if worker_log_path.exists() and 'ReadTimeout' in worker_log_path.read_text(errors='replace'):
            mark_allocation_uncertain(job)
        released = cleanup(api, job)
        if released:
            write_json(job_path.with_suffix(".done"), {"cleanup_verified": True})
        after = usage(api)
        content = worker_log_path.read_text(errors="replace") if worker_log_path.exists() else ""
        smoke = None
        for line in content.splitlines():
            if "SMOKE_RESULT=" in line:
                try:
                    smoke = json.loads(line.split("SMOKE_RESULT=", 1)[1])
                except json.JSONDecodeError:
                    pass
        report = {"job": job["name"], "cli_exit_code": exit_code,
                  "total_wall_seconds": round(time.time() - job["started"], 3),
                  "cleanup_verified": released, "before": before, "after": after,
                  "observed_cu_spent": round(max(0, before["balance_cu"] - after["balance_cu"]), 6),
                  "smoke": smoke, "policy": policy}
        stopped = job_path.with_suffix(".stopped")
        if stopped.exists():
            report["watchdog_stop"] = json.loads(stopped.read_text())
        result_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(result_path, report)
        print(json.dumps(report, indent=2), flush=True)
    return 0 if released and smoke and smoke.get("passed") and exit_code == 0 else 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["status", "plan", "smoke", "recover", "_watch", "_worker"])
    parser.add_argument("job", nargs="?")
    args = parser.parse_args()
    if args.action == "plan":
        print(json.dumps(POLICY, indent=2))
        return 0
    if args.action == "status":
        print(json.dumps(usage(client()), indent=2))
        return 0
    if args.action == "smoke":
        return run_smoke()
    if args.action == "_worker":
        return worker(Path(args.job))
    if args.action == "_watch":
        return watchdog(Path(args.job))
    if args.action == "recover":
        api = client()
        outcomes = []
        for path in PRIVATE.glob("budget-*.json"):
            if path.name.endswith('.allocation-uncertain.json'):
                continue
            job = json.loads(path.read_text())
            released = cleanup(api, job)
            outcomes.append({"job": job["name"], "cleanup_verified": released})
            if released:
                write_json(path.with_suffix(".done"), {"cleanup_verified": True})
        print(json.dumps({"recovery": outcomes, "current": usage(api)}, indent=2))
        return 0 if all(o["cleanup_verified"] for o in outcomes) else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, Exception) as exc:
        # Avoid rich CLI tracebacks exposing cached session credentials.
        print(f"Stopped: {type(exc).__name__}: {str(exc).splitlines()[0][:200]}", file=sys.stderr)
        sys.exit(1)
