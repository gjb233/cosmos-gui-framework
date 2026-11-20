import subprocess,json
from pathlib import Path
r=Path(__file__).resolve().parent
import time,fcntl
lock=(r/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def status(stage):(r/'queue-state.json').write_text(json.dumps(dict(stage=stage,step=250)))
status('waiting_for_lane')
while not (r.parent/'ce-200/verified.json').is_file():time.sleep(20)
assert json.loads((r.parent/'ce-200/exit.json').read_text())['exit_code']==0
status('evaluating')
subprocess.run(['python3',r/'run.py'],cwd=r,check=True)
assert json.loads((r/'exit.json').read_text())['exit_code']==0
for name,n in [('grounding',796),('mind2web',3349)]:
 d=r/name
 def ids(f):return [json.loads(l)['id'] for l in (d/f).open()]
 a=ids('predictions.jsonl');m=ids('manifest.jsonl')
 assert len(a)==len(set(a))==len(m)==n and set(a)==set(m)
 assert json.loads((d/'complete.json').read_text())['checkpoint']==250
(r/'verified.json').write_text((r/'task.json').read_text())

status('verified')
