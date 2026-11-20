from pathlib import Path
import json,time,subprocess,fcntl
g=Path(__file__).resolve().parent
lock=(g/'memory.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
with (g/'device-memory.jsonl').open('a',buffering=1) as f:
 while True:
  try:state=json.loads((g/'state.json').read_text())
  except (ValueError,FileNotFoundError):time.sleep(2);continue
  rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
  f.write(json.dumps(dict(time=time.time(),job=state.get('job'),stage=state.get('stage'),gpus=[r for r in rows.splitlines() if int(r.split(',')[0])<7]))+'\n')
  time.sleep(2)
