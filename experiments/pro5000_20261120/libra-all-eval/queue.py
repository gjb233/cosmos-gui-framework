from pathlib import Path
import subprocess,json,time,fcntl,traceback
R=Path('/data/mai/evaluation/gui-libra-all-20261114');R.mkdir(exist_ok=True,parents=True)
E=Path('/data/mai/libra-all-eval');P='/data/mai/envs/mai/bin/python'
lock=(R/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def status(phase,**kw):
 tmp=R/'status.tmp';tmp.write_text(json.dumps(dict(phase=phase,time=time.time(),**kw)));tmp.replace(R/'status.json')
def run(script,log):
 with log.open('a') as f:subprocess.run([P,'-u',str(script)],stdout=f,stderr=subprocess.STDOUT,check=True)
try:
 old=Path('/data/mai/evaluation/gui-libra-step200-20261114')
 assert json.loads((old/'complete.json').read_text())['completed'],'step200 must finish first'
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied'
 (R/'plan.json').write_text(json.dumps(dict(steps=[0,2,50,100,150,200],reuse_200=str(old),benchmarks=['AndroidControl','Mind2Web','AndroidWorld'],androidworld_seed=30,training_remains_paused=True)))
 for step in (0,2,50,100,150):
  d=R/f'step-{step:06d}';d.mkdir(exist_ok=True);scripts=E/f'step-{step:06d}'
  if (d/'complete.json').exists():continue
  status('preparing_model',step=step)
  if step==0:
   model=d/'models/step-000000';model.parent.mkdir(exist_ok=True)
   if not model.exists():model.symlink_to('/data/mai/benchmarks/gui-libra-base/ready-model',target_is_directory=True)
  else:run(scripts/'merge_eval_models.py',d/'merge.log')
  if not (d/'offline-complete.json').exists():
   status('evaluating_offline',step=step);run(scripts/'run_vllm_eval.py',d/'offline.log')
  if not (d/'androidworld/parallel-complete.json').exists():
   status('evaluating_androidworld',step=step);run(scripts/'androidworld_parallel.py',d/'androidworld.log')
  (d/'complete.json').write_text(json.dumps(dict(completed=True,step=step)))
  summary={}
  for st in (0,2,50,100,150,200):
   p=old if st==200 else R/f'step-{st:06d}'
   if not (p/'complete.json').exists():continue
   summary[str(st)]=dict(offline=json.loads((p/'comparison.json').read_text())[str(st)],androidworld=json.loads((p/'androidworld/summary.json').read_text()))
  (R/'comparison.json').write_text(json.dumps(summary,indent=2))
 status('completed',steps=[0,2,50,100,150,200]);(R/'complete.json').write_text(json.dumps(dict(completed=True)))
except Exception as e:
 status('failed',error=repr(e));traceback.print_exc();raise
