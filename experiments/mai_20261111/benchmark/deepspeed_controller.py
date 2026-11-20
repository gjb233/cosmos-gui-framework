from pathlib import Path
import os,json,time,subprocess,fcntl
g=Path(__file__).resolve().parent
py='/rscratch/wzzheng/code/cosmos-gui/.local/gui-venv/bin/python'
lock=(g/'deepspeed-queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def status(**kw):(g/'deepspeed-queue-state.json').write_text(json.dumps(kw,indent=2))
status(stage='waiting_pytorch_matrix')
while True:
 try:s=json.loads((g/'state.json').read_text())
 except (ValueError,FileNotFoundError):time.sleep(5);continue
 if s['stage']=='initial_matrix_complete_pending_deepspeed_serial':break
 time.sleep(10)
for job in json.loads((g/'deepspeed-plan.json').read_text())['jobs']:
 d=Path(job['path'])
 if (d/'exit.json').exists():
  ex=json.loads((d/'exit.json').read_text())
  if ex==dict(engine=0,producer=0):continue
  status(stage='existing_failure_requires_diagnosis',job=job['name']);raise SystemExit(1)
 status(stage='testing',job=job['name'])
 with (d/'queue-launch.log').open('a') as f:
  p=subprocess.Popen([py,str(d/'dispatch.py')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  (d/'queue-active.json').write_text(json.dumps(dict(pid=p.pid)))
  rc=p.wait()
 ex=json.loads((d/'exit.json').read_text()) if (d/'exit.json').exists() else {}
 if rc or ex!=dict(engine=0,producer=0):
  status(stage='failed_requires_diagnosis',job=job['name'],launcher=rc,exit=ex);raise SystemExit(1)
 status(stage='finished_unverified',job=job['name'])
status(stage='deepspeed_runs_complete_require_audit_serial_pending')
