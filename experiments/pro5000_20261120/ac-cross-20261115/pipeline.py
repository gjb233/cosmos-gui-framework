from pathlib import Path
import subprocess,os,json,time,fcntl
B=Path('/data/mai/ac-cross-20261115');R=Path('/data/mai/evaluation/ac-cross-20261115')
lock=(R/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def status(**x):(R/'status.json').write_text(json.dumps(dict(time=time.time(),**x)))
try:
 for arm in ['official','ac-high']:
  status(phase='running',benchmark='mind2web',arm=arm)
  with (R/f'mind2web-{arm}.log').open('a') as f:subprocess.run(['/data/mai/envs/mai/bin/python','-u',str(B/'web_run.py')],env=dict(os.environ,PROMPT_ARM=arm),stdout=f,stderr=subprocess.STDOUT,check=True)
 status(phase='running',benchmark='androidworld',arm='ac-high')
 with (R/'androidworld.log').open('a') as f:subprocess.run(['/data/mai/envs/mai/bin/python','-u',str(B/'launch.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
 status(phase='completed')
except Exception as e:status(phase='failed',error=repr(e));raise
