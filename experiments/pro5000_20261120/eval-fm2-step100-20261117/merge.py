from pathlib import Path
import torch,json,shutil,hashlib,concurrent.futures
from safetensors.torch import load_file,save_file
S=Path('/data/mai/benchmarks/gui-libra-base/ready-model')
R=Path('/data/mai/evaluation/fm2-step100-20261117/direct');R.mkdir(parents=True,exist_ok=True)
T=Path('/data/mai/training/gui-libra-awhigh-aclow-fm2-20261117/run/ar-adapters')
torch.set_num_threads(8)
def merge(step):
 out=R/'models'/f'step-{step:06d}';out.mkdir(parents=True,exist_ok=True)
 if (out/'merge-verified.json').exists():return
 adapter=T/f'ar-step-{step:06d}.safetensors';state=load_file(str(adapter));assert len(state)==504
 delta={}
 for key,a in state.items():
  if '.lora_A.weight' not in key:continue
  b=state[key.replace('.lora_A.weight','.lora_B.weight')]
  weight=key.replace('language_model.model.','model.language_model.').replace('.lora_A.weight','.weight')
  delta[weight]=(a,b)
 assert len(delta)==252
 for p in S.iterdir():
  if p.is_file() and p.suffix!='.safetensors':shutil.copy2(p,out/p.name)
 index=json.loads((S/'model.safetensors.index.json').read_text())['weight_map']
 assert set(delta)<=set(index)
 audit=[]
 for filename in sorted(set(index.values())):
  values=load_file(str(S/filename))
  for key in set(values)&set(delta):
   a,b=delta[key];original=values[key]
   update=b.float()@a.float()*2
   assert update.shape==original.shape and torch.isfinite(update).all()
   merged=(original.float()+update).to(original.dtype);assert torch.isfinite(merged).all()
   audit.append(dict(key=key,max_delta=float(update.abs().max()),changed_elements=int((merged!=original).sum())))
   values[key]=merged
  tmp=out/(filename+'.tmp');save_file(values,str(tmp),metadata={'format':'pt'});tmp.replace(out/filename)
  del values
 assert len(audit)==252 and sum('.mlp.' in x['key'] for x in audit)==108
 assert all(x['changed_elements']>0 for x in audit)
 report=dict(step=step,adapter_sha256=hashlib.sha256(adapter.read_bytes()).hexdigest(),merged_weight_matrices=252,mlp_matrices=108,attention_matrices=144,lora_rank=16,lora_alpha=32,merge_dtype='FP32 addition; preserved base dtype',verified=True,weights=audit)
 (out/'merge-verified.json').write_text(json.dumps(report,indent=2));print('MERGE_VERIFIED',step,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:list(pool.map(merge,(100,)))
