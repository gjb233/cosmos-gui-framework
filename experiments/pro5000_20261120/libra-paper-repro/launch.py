from pathlib import Path
import os,json,subprocess,time,fcntl
R=Path('/data/mai/evaluation/gui-libra-paper-repro-20261114');R.mkdir(exist_ok=True,parents=True)
lock=(R/'launch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
 for n in (1,2):
  (R/'status.json').write_text(json.dumps(dict(phase='running',round=n,time=time.time())))
  with (R/f'round-{n}.log').open('a') as f:subprocess.run(['/data/mai/envs/mai/bin/python','-u','/data/mai/libra-paper-repro/parallel.py'],env=dict(os.environ,LIBRA_REPRO_ROUND=str(n)),stdout=f,stderr=subprocess.STDOUT,check=True)
 results=[json.loads((R/f'round-{n}/summary.json').read_text()) for n in (1,2)]
 (R/'complete.json').write_text(json.dumps(dict(completed=True,rounds=results,official_code_mean=sum(r['official_code_successes']/115 for r in results)/2,paper_terminate_mean=sum(r['paper_terminate_successes']/115 for r in results)/2),indent=2))
 (R/'status.json').write_text(json.dumps(dict(phase='completed',time=time.time())))
except Exception as e:
 (R/'status.json').write_text(json.dumps(dict(phase='failed',error=repr(e),time=time.time())));raise
