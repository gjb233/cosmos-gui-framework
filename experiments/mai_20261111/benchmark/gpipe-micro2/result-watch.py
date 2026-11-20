from pathlib import Path
import time,json,subprocess,fcntl,os
o=Path(__file__).resolve().parent
py='/rscratch/wzzheng/code/cosmos-gui/.local/gui-venv/bin/python'
lock=(o/'result-watch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def state(**kw):
 t=o/'result-watch-state.tmp';t.write_text(json.dumps(dict(time=time.time(),**kw),indent=2));t.replace(o/'result-watch-state.json')
while True:
 for ready in sorted((o/'checkpoints').glob('step-*/ready.json')):
  cp=ready.parent
  if (cp/'cpu-verified.json').exists():continue
  step=int(cp.name.split('-')[-1])
  with (o/'checkpoint-audit.log').open('a') as f:
   rc=subprocess.run([py,str(o/'verify-checkpoint.py'),str(step)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT).returncode
  if rc:state(stage='checkpoint_verification_failed',step=step,exit_code=rc);raise SystemExit(rc)
  state(stage='checkpoint_verified',step=step)
 if (o/'checkpoints/step-000354/cpu-verified.json').exists() and not (o/'benchmark-result.json').exists():
  completed=True
  for r in range(7):
   p=o/f'saves-rank{r}.jsonl'
   if not p.exists() or not any(json.loads(l)['step']==354 for l in p.read_text().splitlines() if l.strip()):completed=False
  if completed:
   with (o/'comparison.log').open('a') as f:
    rc=subprocess.run([py,str(o/'compare.py')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT).returncode
   if rc:state(stage='comparison_failed',exit_code=rc);raise SystemExit(rc)
   state(stage='comparison_complete',result=str(o/'benchmark-result.json'))
 if (o/'exit.json').exists():
  state(stage='training_exited',exit=json.loads((o/'exit.json').read_text()));break
 if not (o/'active.json').exists():
  d=json.loads((o/'dispatcher-active.json').read_text())
  if not Path(f"/proc/{d['pid']}").exists():
   state(stage='dispatcher_exited_before_launch');break
 time.sleep(15)
