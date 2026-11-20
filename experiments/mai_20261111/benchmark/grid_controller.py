from pathlib import Path
import json,time,os,signal,subprocess,fcntl
G=Path(__file__).resolve().parent;O=G.parent/'mai-ui-70283-resume350-1f1b7-micro4-gb56-a22'
PY='/rscratch/wzzheng/code/cosmos-gui/.local/gui-venv/bin/python'
lock=(G/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def state(**kw):(G/'state.json').write_text(json.dumps(kw,indent=2))
state(stage='waiting_current_354_verified')
while not (O/'checkpoints/step-000354/cpu-verified.json').exists():
 if (O/'exit.json').exists():state(stage='current_failed_before_verified');raise SystemExit(1)
 time.sleep(5)
while not (O/'benchmark-result.json').exists():time.sleep(5)
a=json.loads((O/'active.json').read_text());pid=a['engine_pid']
if Path(f'/proc/{pid}').exists():
 assert str(O/'driver.py') in Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\x00',b' ').decode()
 os.killpg(pid,signal.SIGTERM)
while not (O/'exit.json').exists():time.sleep(3)
(G/'paused-after354.json').write_text(json.dumps(dict(checkpoint=str(O/'checkpoints/step-000354'),preserved=True)))
for j in json.loads((G/'plan.json').read_text())['jobs']:
 d=Path(j['path'])
 if (d/'queue-exit.json').exists():continue
 state(stage='testing',job=j['name'])
 with (d/'queue-launch.log').open('a') as f:
  p=subprocess.Popen([PY,str(d/j['launch'])],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  (d/'queue-active.json').write_text(json.dumps(dict(pid=p.pid)))
  rc=p.wait()
 (d/'queue-exit.json').write_text(json.dumps(dict(exit_code=rc)))
 time.sleep(10)
state(stage='initial_matrix_complete_pending_deepspeed_serial')
