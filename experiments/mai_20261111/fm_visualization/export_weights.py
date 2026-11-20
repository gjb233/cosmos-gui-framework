from pathlib import Path
import json,torch
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader,load
torch.set_num_threads(2)
o=Path(__file__).resolve().parent;p=o.parent
ck=p/'mai-ui-70284-input16384-1f1b-micro4-gb64-a21-gpu17-save50/1f1b/checkpoints/step-000100'
w={}
for rank in [0,1]:
 s=torch.load(ck/f'rank{rank}.pt',map_location='cpu',weights_only=True)
 assert s['step']==100
 for name,v in s['params'].items():
  if name.startswith('endpoint.'):key=name.removeprefix('endpoint.')
  else:
   layer,tail=name.split('decoder.flat_stage.decoder.layers.')[1].split('.',1)
   key=f'language_model.model.layers.{int(layer)+17*rank}.{tail}'
  assert key not in w
  w[key]=v.contiguous()
assert len(w)==584
save_file(w,str(o/'weights-100.safetensors'));print('EXPORTED100',len(w),flush=True)
root=p/'mai-ui-70283-resume150-fsdp7-micro4-gb56-a22-fullmem/runs/20261110T090029Z/runtime'
for step in [200,300]:
 d=next(root.rglob(f'iter_{step:09d}'))/'model'
 reader=FileSystemReader(d);m=reader.read_metadata()
 state={}
 for k,v in w.items():
  name='net.'+k
  assert name in m.state_dict_metadata,name
  spec=m.state_dict_metadata[name];assert tuple(spec.size)==tuple(v.shape)
  state[name]=torch.empty(spec.size,dtype=spec.properties.dtype)
 load(state,storage_reader=reader)
 vals={k.removeprefix('net.'):v for k,v in state.items()}
 assert all(torch.isfinite(v).all() for v in vals.values())
 save_file(vals,str(o/f'weights-{step}.safetensors'))
 print('EXPORTED',step,len(vals),flush=True)
