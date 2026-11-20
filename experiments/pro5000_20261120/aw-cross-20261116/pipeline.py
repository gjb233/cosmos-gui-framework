from pathlib import Path
import subprocess,os,json,time,fcntl
B=Path('/data/mai/aw-cross-20261116');R=Path('/data/mai/evaluation/aw-cross-20261116')
lock=(R/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def status(**x):(R/'status.json').write_text(json.dumps(dict(time=time.time(),**x)))
try:
 for arm in ['official','aw-core','aw-full']:
  status(phase='running',arm=arm)
  with (R/f'{arm}.log').open('a') as f:subprocess.run(['/data/mai/envs/mai/bin/python','-u',str(B/'web_run.py')],env=dict(os.environ,PROMPT_ARM=arm),stdout=f,stderr=subprocess.STDOUT,check=True)
 status(phase='completed')
except Exception as e:status(phase='failed',error=repr(e));raise
