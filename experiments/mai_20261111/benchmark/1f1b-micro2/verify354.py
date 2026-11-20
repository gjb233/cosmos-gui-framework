from pathlib import Path
import torch,json,math
from safetensors.torch import load_file
o=Path(__file__).resolve().parent
p=o/'checkpoints/step-000354'
ready=json.loads((p/'ready.json').read_text());assert ready['step']==354 and ready['next_draw']==21024
source=torch.load(o/'migration-350.pt',map_location='cpu',weights_only=True)
adapter=load_file(str(p/'ar-adapter.safetensors'));assert len(adapter)==288
seen=set();changed={'ar':0,'fm':0,'other':0};ranks=[]
for r in range(7):
 ck=torch.load(p/f'rank{r}.pt',map_location='cpu',weights_only=True)
 assert ck['step']==354 and ck['next_draw']==21024
 assert ck['scheduler']['updates_since350']==4
 expected=1e-5*(3298-(150+204*56/64))/(3298-33)
 assert math.isclose(ck['scheduler']['lr'],expected,rel_tol=1e-12)
 assert len(ck['rng'])>0 and len(ck['cuda_rng'])>0
 assert set(ck['params'])==set(ck['optimizer'])==set(ck['groups'])
 for n,v in ck['params'].items():
  assert n not in seen;seen.add(n)
  assert torch.isfinite(v).all()
  state=ck['optimizer'][n];assert int(state['step'])==354
  assert all(torch.isfinite(t).all() for t in state.values() if isinstance(t,torch.Tensor))
  branch='fm' if '_moe_gen' in n else 'ar' if 'lora_' in n else 'other'
  changed[branch]+=int(not torch.equal(v,source['params'][n]))
  if n in adapter:assert torch.equal(v,adapter[n])
 ranks.append({'rank':r,'parameters':len(ck['params']),'optimizer_step':354})
assert seen==set(source['params']) and len(seen)==584
assert changed['ar']==288 and changed['fm']>0 and changed['other']>0
rng=torch.load(p/'producer-rng.pt',map_location='cpu',weights_only=True)
assert rng['step']==354 and rng['next_draw']==21024 and len(rng['rng_states'])==7
result=dict(passed=True,step=354,next_draw=21024,parameters=584,adapter_tensors=288,changed=changed,ranks=ranks,producer_rng_streams=7)
(p/'cpu-verified.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
