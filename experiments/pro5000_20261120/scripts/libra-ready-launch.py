from pathlib import Path
import hashlib,json,time,subprocess,fcntl,os
base=Path('/data/mai/benchmarks/gui-libra-base')
w=Path('/data/mai/training/gui-libra-zero2-mlp-strict10k-20261114')
lock=(w/'ready-launch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (w/'run/active.json').exists()
assert json.loads((w/'input-validation.json').read_text())['passed']
assert json.loads((w/'processor-smoke.json').read_text())['passed']
expected=['05730ab0b20b817b166965510d77cc04fdceecfca7ed07bf14de4f6995b58b79','532a7459642847e41d1816049251c0846710610bb3b01dcec4d180bd3e4c9043','580d7849fead7b42edb0d63ae2ef6ab14ae6fa99dca601a9f452274f33f12014','eeafdce81f0546ca4a7bb645f0f27013d15f1519ad3d280f9bf54305565444dd']
ready=base/'ready-model';ready.mkdir(exist_ok=True)
for p in (base/'model').iterdir():
 if p.is_file() and p.suffix in ('.json','.txt','.jinja'):
  if not (ready/p.name).exists():(ready/p.name).symlink_to(p)
verified={};deadline=time.monotonic()+4*3600
while len(verified)<4:
 assert time.monotonic()<deadline,'weight transfer timed out'
 for i,sha in enumerate(expected,1):
  name=f'model-{i:05d}-of-00004.safetensors'
  if name in verified:continue
  for directory in ('model','cached-model','mirror-model'):
   p=base/directory/name
   if not p.exists():continue
   h=hashlib.sha256()
   with p.open('rb') as f:
    for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
   assert h.hexdigest()==sha,(name,'checksum mismatch')
   if not (ready/name).exists():(ready/name).symlink_to(p)
   verified[name]=dict(sha256=sha,path=str(p));print('WEIGHT_VERIFIED',name,directory,flush=True);break
 (w/'weight-readiness.json').write_text(json.dumps(dict(verified=verified,complete=len(verified)==4)))
 if len(verified)<4:time.sleep(15)
index=json.loads((ready/'model.safetensors.index.json').read_text())
assert set(index['weight_map'].values())==set(verified)
for name in ('training_backend.py','patch_training.py'):
 p=w/name;p.write_text(p.read_text().replace(str(base/'model'),str(ready)))
p=w/'train_entry.py';p.write_text(p.read_text().replace('MAI_ORIGINAL_RES_GPU_BUDGET','GUI_LIBRA_ORIGINAL_RES_GPU_BUDGET').replace('MAI native extended mobile_use shared renderer','GUI-Libra native answer renderer with full original action history'))
p=w/'training-plan.json';plan=json.loads(p.read_text());plan['initial_ar_path']=str(ready);plan['initial_ar_weights']=verified;plan['fresh_optimizer']=True;plan['fresh_lora']=True;p.write_text(json.dumps(plan,indent=2))
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not gpu,'GPU occupied; refusing to start alongside another job'
assert not (w/'run/checkpoints/latest-complete.json').exists()
print('LAUNCHING_FRESH_GUI_LIBRA',flush=True)
subprocess.run(['/data/mai/envs/mai/bin/python','-u','/data/mai/launch_libra_fulltrain.py'],check=True)
