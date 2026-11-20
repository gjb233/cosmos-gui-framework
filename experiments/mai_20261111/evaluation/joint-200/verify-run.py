import subprocess,json
from pathlib import Path
r=Path(__file__).resolve().parent
subprocess.run(['python3',r/'run.py'],cwd=r,check=True)
assert json.loads((r/'exit.json').read_text())['exit_code']==0
for name,n in [('grounding',796),('mind2web',3349)]:
 d=r/name
 def ids(f):return [json.loads(l)['id'] for l in (d/f).open()]
 a=ids('predictions.jsonl');m=ids('manifest.jsonl')
 assert len(a)==len(set(a))==len(m)==n and set(a)==set(m)
 assert json.loads((d/'complete.json').read_text())['checkpoint']==200
(r/'verified.json').write_text((r/'task.json').read_text())
