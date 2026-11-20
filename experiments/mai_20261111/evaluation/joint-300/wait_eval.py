import subprocess,json
from pathlib import Path
r=Path(__file__).resolve().parent
import time,fcntl
lock=(r/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def status(stage):(r/'queue-state.json').write_text(json.dumps(dict(stage=stage,step=300)))
import torch,hashlib
from safetensors.torch import load_file,save_file
torch.set_num_threads(2)
task=json.loads((r/'task.json').read_text())
source=Path(task['weights'])
ck=next((source.parents[1]/'runtime').rglob('iter_000000275')).with_name('iter_000000300')
def complete():
 for n in ['model','optim','trainer','scheduler']:
  d=ck/n
  if not (d/'.metadata').is_file():return False
  if not all((d/f'__{i}_0.distcp').is_file() for i in range(7)):return False
 return source.is_file() and source.with_suffix('.ready.json').is_file()
status('waiting_for_checkpoint')
while not complete():time.sleep(20)
assert json.loads(source.with_suffix('.ready.json').read_text())['iteration']==300
w=load_file(str(source));ref=load_file(str(r.parent/'joint-250/adapter/adapter_model.safetensors'))
v={k.replace('_checkpoint_wrapped_module.','').replace('language_model.model.layers.','base_model.model.model.language_model.layers.'):t for k,t in w.items()}
assert len(w)==288 and set(v)==set(ref)
assert all(t.shape==ref[k].shape and torch.isfinite(t).all() for k,t in v.items())
out=r/'adapter/adapter_model.safetensors'
save_file({k:t.to(ref[k].dtype) for k,t in v.items()},str(out))
assert all(torch.equal(t, v[k]) for k,t in load_file(str(out)).items())
task.update(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),export_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),tensors=288)
(r/'task.json').write_text(json.dumps(task))
status('waiting_for_lane')
while not (r.parent/'joint-250/verified.json').is_file():time.sleep(20)
assert json.loads((r.parent/'joint-250/exit.json').read_text())['exit_code']==0
status('evaluating')
subprocess.run(['python3',r/'run.py'],cwd=r,check=True)
assert json.loads((r/'exit.json').read_text())['exit_code']==0
for name,n in [('grounding',796),('mind2web',3349)]:
 d=r/name
 def ids(f):return [json.loads(l)['id'] for l in (d/f).open()]
 a=ids('predictions.jsonl');m=ids('manifest.jsonl')
 assert len(a)==len(set(a))==len(m)==n and set(a)==set(m)
 assert json.loads((d/'complete.json').read_text())['checkpoint']==300
(r/'verified.json').write_text((r/'task.json').read_text())

status('verified')
