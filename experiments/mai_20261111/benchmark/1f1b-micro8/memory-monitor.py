from pathlib import Path
import subprocess,time,json,os,fcntl
o=Path(__file__).resolve().parent
lock=(o/'memory-monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
with (o/'device-memory.csv').open('a',buffering=1) as f:
 while not (o/'exit.json').exists():
  rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
  ts=time.time()
  for row in rows.splitlines():f.write(str(ts)+','+row+'\n')
  time.sleep(2)
