import json,sys,time,subprocess,hashlib,fcntl
from pathlib import Path
from safetensors.torch import load_file,save_file
import torch
torch.set_num_threads(2)
b=Path(__file__).resolve().parent
step=int(sys.argv[1]);tag=f'ce-{step}';r=b/tag
lock=(r/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
cfg=json.loads((r/'task.json').read_text())
ck=Path(cfg['checkpoint_dir'])
def status(stage): (r/'queue-state.json').write_text(json.dumps(dict(stage=stage,step=step,time=time.time())))
status('waiting_for_checkpoint')
while not (ck/'trainer_state.json').exists():time.sleep(30)
assert json.loads((ck/'trainer_state.json').read_text())['global_step']==step
source=ck/'adapter_model.safetensors'; w=load_file(str(source));ref=load_file(str(b/'ce-50/adapter/adapter_model.safetensors'))
assert len(w)==288 and set(w)==set(ref)
weights={}
for k,v in w.items():
 assert v.shape==ref[k].shape and torch.isfinite(v).all()
 weights[k]=v.to(ref[k].dtype)
 assert torch.isfinite(weights[k]).all()
out=r/'adapter/adapter_model.safetensors';save_file(weights,str(out.with_suffix('.tmp')));out.with_suffix('.tmp').replace(out)
def digest(f):return hashlib.sha256(f.read_bytes()).hexdigest()
(r/'adapter-source.json').write_text(json.dumps(dict(source=str(source),source_sha256=digest(source),export_sha256=digest(out),step=step,count=len(w))))
status('waiting_for_lane')
while not (b/cfg['after']/'verified.json').is_file():time.sleep(20)
status('evaluating')
with (r/'controller.log').open('x') as f:subprocess.run(['python3',r/'run.py'],cwd=r,stdout=f,stderr=subprocess.STDOUT,check=True)
assert json.loads((r/'exit.json').read_text())['exit_code']==0
for name,count in [('grounding',796),('mind2web',3349)]:
 def ids(f):return [json.loads(l)['id'] for l in (r/name/f).open()]
 expected=ids('manifest.jsonl');actual=ids('predictions.jsonl')
 assert len(expected)==len(actual)==len(set(actual))==count and set(expected)==set(actual)
 assert json.loads((r/name/'complete.json').read_text())['checkpoint']==step
(r/'verified.json').write_text(json.dumps(cfg));status('verified')
