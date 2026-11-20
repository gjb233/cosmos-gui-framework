from pathlib import Path
import subprocess,os,json,time,signal,fcntl
B=Path('/data/mai/aw-fm2-step100-20261118');R=Path('/data/mai/evaluation/aw-fm2-step100-20261118');W=Path('/data/mai/training/gui-libra-awhigh-aclow-fm2-20261117')
lock=(R/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def status(phase,**kw):(R/'status.json').write_text(json.dumps(dict(phase=phase,time=time.time(),**kw)))
try:
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied'
 status('evaluating')
 with (R/'parallel.log').open('a') as f:subprocess.run(['/data/mai/envs/mai/bin/python','-u',str(B/'androidworld_parallel.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
 status('completed',summary=json.loads((R/'summary.json').read_text()))
except Exception as e:status('failed',error=repr(e));raise
finally:
 # Resume only after all evaluation GPU processes have exited.
 deadline=time.time()+180
 while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip() and time.time()<deadline:time.sleep(5)
 if not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
  p=subprocess.Popen(['/data/mai/envs/mai/bin/python','-u',str(W/'launch.py')],stdout=(W/'run/resume-after-androidworld100.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
  (R/'training-resume.json').write_text(json.dumps(dict(pid=p.pid,time=time.time())))
