from pathlib import Path
import os,json,time,signal,subprocess,fcntl,psutil,traceback
Q=Path('/data/mai/step200-suite-20261118');R=Path('/data/mai/evaluation/step200-suite-20261118');W=Path('/data/mai/training/gui-libra-awhigh-aclow-fm2-20261117');PY='/data/mai/envs/mai/bin/python'
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
 assert not (R/'complete.json').exists(),'already completed'
 state('waiting_checkpoint',target_step=200)
 while True:
  a=json.loads((W/'run/active.json').read_text());pid=a['pid'];cmd=Path(f'/proc/{pid}/cmdline')
  assert cmd.exists() and str(W/'train_entry.py').encode() in cmd.read_bytes(),'current training stopped before step200'
  saved=json.loads((W/'run/checkpoints/latest-complete.json').read_text())
  if saved['completed_steps']>=200:break
  (R/'watch.json').write_text(json.dumps(dict(time=time.time(),pid=pid,saved_step=saved['completed_steps'],target_step=200)))
  time.sleep(5)
 checkpoint=W/'run/checkpoints/step-000200'
 assert len(list(checkpoint.glob('*optim_states.pt')))==8
 assert len(list(checkpoint.glob('rng-rank*.pt')))==8
 assert (W/'run/ar-adapters/ar-step-000200.safetensors').exists()
 (R/'pause.json').write_text(json.dumps(dict(pid=pid,resume_step=saved['completed_steps'],target_step=200)))
 state('pausing_at_checkpoint',saved_step=saved['completed_steps']);os.killpg(os.getpgid(pid),signal.SIGTERM);paused=True;wait_gpu()
 merged=stage('fm2_merge200',Q/'fm2-offline200/merge.py',R/'fm2-offline200/direct/models/step-000200/merge-verified.json')
 if merged:
  stage('fm2_offline200',Q/'fm2-offline200/web_run.py',R/'fm2-offline200/direct/offline-complete.json')
  stage('fm2_androidworld200',Q/'fm2-aw200/androidworld_parallel.py',R/'fm2-aw200/parallel-complete.json')
 stage('fm_visualization200',Q/'fm-visual/launch.py',R/'fm-visual/complete.json',timeout=7200)
 (R/'queue-results.json').write_text(json.dumps(dict(finished=True,errors=errors),indent=2))
 if not errors:(R/'complete.json').write_text(json.dumps(dict(completed=True)))
 state('completed_with_errors' if errors else 'completed')
except Exception as e:
 state('failed',error=repr(e));traceback.print_exc()
finally:
 if paused:
  try:
   wait_gpu()
   p=subprocess.Popen([PY,'-u',str(W/'launch.py')],stdin=subprocess.DEVNULL,stdout=(W/'run/resume-after-step200-suite.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
   (R/'training-resume.json').write_text(json.dumps(dict(pid=p.pid,time=time.time())))
  except Exception as e:(R/'resume-failed.json').write_text(json.dumps(dict(error=repr(e))))
