from pathlib import Path
import os,json,time,signal,subprocess,fcntl,psutil,traceback
Q=Path('/data/mai/libra-fm2-eval50-150-20261118');R=Path('/data/mai/evaluation/libra-fm2-eval50-150-20261118');W=Path('/data/mai/training/gui-libra-awhigh-aclow-fm2-20261117');PY='/data/mai/envs/mai/bin/python'
lock=(R/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
errors=[];paused=False

def state(phase,**kw):
 p=R/'status.tmp';p.write_text(json.dumps(dict(phase=phase,time=time.time(),errors=errors,**kw)));p.replace(R/'status.json')
def gpu_free():return not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
def wait_gpu(timeout=240):
 deadline=time.time()+timeout
 while not gpu_free():
  assert time.time()<deadline,'GPU release timeout';time.sleep(3)
def stage(name,script,marker,timeout=21600):
 if marker.exists():return True
 state('running_stage',stage=name)
 processes={}
 with (R/(name+'.log')).open('a') as log:
  p=subprocess.Popen([PY,'-u',str(script)],env=dict(os.environ,PROMPT_ARM='direct'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  (R/'active-stage.json').write_text(json.dumps(dict(stage=name,pid=p.pid,script=str(script))))
  deadline=time.time()+timeout
  while p.poll() is None and time.time()<deadline:
   try:
    for c in [psutil.Process(p.pid),*psutil.Process(p.pid).children(recursive=True)]:processes[c.pid]=c.create_time()
   except psutil.NoSuchProcess:pass
   time.sleep(2)
  timed_out=p.poll() is None
  if timed_out:os.killpg(p.pid,signal.SIGTERM)
  # Stop only verified descendants of this stage; servers use separate sessions.
  if timed_out or p.returncode not in (0,None):
   for pid,created in processes.items():
    try:
     c=psutil.Process(pid)
     if c.create_time()==created:c.terminate()
    except psutil.NoSuchProcess:pass
  if timed_out:
   try:p.wait(timeout=60)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  wait_gpu()
  ok=p.returncode==0 and marker.exists()
  if not ok:errors.append(dict(stage=name,exit_code=p.returncode,timeout=timed_out,missing_marker=not marker.exists()))
  return ok
try:
 assert gpu_free(), 'GPU still occupied'
 assert stage('mai_androidworld200_repair',Q/'mai200-repair/run.py',Path('/data/mai/evaluation/step200-suite-20261118/mai-aw200/parallel-complete.json')), 'MAI priority stage failed; GUI remains queued'
 state('merging')
 merged=stage('merge50_150',Q/'offline/merge.py',R/'offline/direct/models/step-000150/merge-verified.json')
 if merged:
  stage('offline50_150',Q/'offline/web_run.py',R/'offline/direct/offline-complete.json')
  for step in [50,150]:
   stage(f'androidworld{step}',Q/f'aw{step}/androidworld_parallel.py',R/f'aw{step}/parallel-complete.json')
 (R/'complete.json').write_text(json.dumps(dict(completed=not errors,errors=errors,training_remains_paused=True)))
 state('completed_with_errors' if errors else 'completed')
except Exception as e:
 state('failed',error=repr(e));traceback.print_exc()
