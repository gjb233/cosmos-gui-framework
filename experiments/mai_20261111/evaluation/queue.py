import sys,subprocess,json
from pathlib import Path
b=Path(__file__).resolve().parent
for task in json.loads((b/'plan.json').read_text()):
 if task['lane']!=sys.argv[1]:continue
 d=b/task['tag']
 with (d/'controller.log').open('x') as f:subprocess.run(['python3',d/'run.py'],cwd=d,stdout=f,stderr=subprocess.STDOUT,check=True)
 for name,n in [('grounding',796),('mind2web',3349)]:
  def ids(file):return [json.loads(l)['id'] for l in (d/name/file).open()]
  a=ids('predictions.jsonl');e=ids('manifest.jsonl');assert len(set(a))==len(a)==len(e)==n and set(a)==set(e)
 (d/'verified.json').write_text(json.dumps(task))
