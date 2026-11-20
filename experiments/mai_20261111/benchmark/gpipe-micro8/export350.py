from pathlib import Path
import json,time,torch,hashlib
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
O=Path(__file__).resolve().parent
while not (O/'paused-350.json').exists():time.sleep(5)
ck=Path(json.loads((O/'paused-350.json').read_text())['checkpoint'])
def read(k,keys=None):
 r=dcp.FileSystemReader(str(ck/k));md=r.read_metadata()
 sd={n:(torch.empty(v.size,dtype=v.properties.dtype) if isinstance(v,TensorStorageMetadata) else None) for n,v in md.state_dict_metadata.items() if keys is None or n in keys}
 dcp.load(sd,storage_reader=r);return sd
optim=read('optim')
names=sorted(k[len('state.net.'):-len('.step')] for k in optim if k.startswith('state.net.') and k.endswith('.step'))
assert len(names)==584
md=dcp.FileSystemReader(str(ck/'model')).read_metadata()
probes=[k for k in md.state_dict_metadata if k.startswith('net.language_model.model.layers.') and 'q_proj' in k and 'lora_' not in k and k.endswith('.weight')]
weights=read('model',set(['net.'+n for n in names]+probes))
params={n:weights['net.'+n] for n in names}
states={n:{q:optim['state.net.'+n+'.'+q] for q in ['exp_avg','exp_avg_sq','step']} for n in names}
groups={n:{q:optim['param_groups.net.'+n+'.'+q] for q in ['lr','betas','eps','weight_decay']} for n in names}
for n in names:
 assert torch.isfinite(params[n]).all()
 assert int(states[n]['step'])==350
 assert all(torch.isfinite(states[n][q]).all() for q in ['exp_avg','exp_avg_sq'])
 assert params[n].shape==states[n]['exp_avg'].shape==states[n]['exp_avg_sq'].shape
trainer=read('trainer');scheduler=read('scheduler');assert trainer['iteration']==350
probe_hash={k.removeprefix('net.'):hashlib.sha256(weights[k].to(torch.bfloat16).contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for k in probes}
out=dict(step=350,next_draw=20800,params=params,optimizer=states,groups=groups,scheduler=scheduler,trainer=trainer,frozen_probe_hashes=probe_hash,source=str(ck))
torch.save(out,O/'migration-350.tmp');(O/'migration-350.tmp').replace(O/'migration-350.pt')
(O/'migration-350.json').write_text(json.dumps(dict(step=350,next_draw=20800,finite_parameters=len(names),finite_adam_states=len(states),rng_ranks=7,scheduler={k:str(v) for k,v in scheduler.items()},frozen_probes=len(probes),source=str(ck)),indent=2))
print('MIGRATED_350',len(names),len(probes),flush=True)
