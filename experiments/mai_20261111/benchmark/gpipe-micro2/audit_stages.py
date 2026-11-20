import os,sys,json,hashlib,torch
from pathlib import Path
O=Path(__file__).resolve().parent
R=O.parent/'mai-framework-comparison-a22-seven-20261107'
sys.path[:0]=[str(R/'torch-pipeline'),'/rscratch/wzzheng/experiments/gui-grounding-fixed-ab-20261002-v1/source']
m=torch.load(O/'migration-350.pt',map_location='cpu',weights_only=True)
names=[];report=[]
for rank in range(7):
 stage=torch.load(R/f'real-stage-reference/reference/stage-{rank}.pt',map_location='cpu',weights_only=False)
 count=0;probes=0
 for local,layer in enumerate(stage.layers):
  for key,p in layer.named_parameters():
   name=f'language_model.model.layers.{stage.start+local}.{key}'
   if p.requires_grad:
    assert name in m['params'] and p.shape==m['params'][name].shape
    names.append(name);count+=1
   if name in m['frozen_probe_hashes']:
    digest=hashlib.sha256(p.detach().to(torch.bfloat16).contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    assert digest==m['frozen_probe_hashes'][name],name
    probes+=1
 report.append(dict(rank=rank,start=stage.start,layers=len(stage.layers),trainable=count,frozen_probe_count=probes))
 print(report[-1],flush=True);del stage
assert len(names)==576 and len(set(names))==576
missing=sorted(set(m['params'])-set(names));assert len(missing)==8
(O/'cpu-stage-audit.json').write_text(json.dumps(dict(stages=report,endpoint_parameters=missing,passed=True),indent=2))
