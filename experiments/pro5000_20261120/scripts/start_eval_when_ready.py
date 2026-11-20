from pathlib import Path
import subprocess,json,time,fcntl,os
R=Path('/data/mai/evaluation/ckpts-through-200-20261113');R.mkdir(parents=True,exist_ok=True)
lock=(R/'prepare.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
archive=Path('/data/mai/archives/pro5000-eval-assets-20261113.tar.gz')
A=Path('/data/mai/evaluation/assets-20261113')
try:
 while not archive.exists():time.sleep(5)
 assert Path('/data/mai/training/mai-ui-zero2-mlp-strict10k-20261113/run/pause-request-200.json').exists()
 A.mkdir(parents=True,exist_ok=True)
 subprocess.run(['tar','-xzf',str(archive),'-C',str(A)],check=True)
 subprocess.run(['tar','-xzf','/data/mai/archives/pro5000-eval-extras-20261113.tar.gz','-C',str(A)],check=True)
 for script in ('prepare_eval_runtime.py','validate_eval_assets.py'):
  subprocess.run(['/data/mai/envs/mai/bin/python','/data/mai/'+script],check=True)
 subprocess.run(['/data/mai/envs/mai/bin/python','/data/mai/run_eval_queue.py'],check=True)
except Exception as e:
 (R/'prepare-failed.json').write_text(json.dumps(dict(error=repr(e))));raise
