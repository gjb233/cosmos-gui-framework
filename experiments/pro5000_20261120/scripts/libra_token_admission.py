from pathlib import Path
import sys,json,hashlib,math
from PIL import Image
from transformers import AutoTokenizer
W=Path('/data/mai/training/gui-libra-zero2-mlp-strict10k-20261114');sys.path.insert(0,str(W))
from native_protocol import render_policy
tok=AutoTokenizer.from_pretrained('/data/mai/benchmarks/gui-libra-base/model',local_files_only=True)
src=Path('/data/mai/assets/training-ready-strict-10k-20261113/train-manifest.jsonl')
keep=[];excluded=[];maxlen=0
for index,line in enumerate(src.open()):
 r=json.loads(line)
 with Image.open(r['current_image']) as im:w,h=im.size
 visual=round(w/32)*round(h/32)
 lengths=[]
 for level,v in r['views'].items():
  policy=dict(current_image=None,image_size=(w,h),instruction=v['instruction'],history=r['history_mobile_use'],level=level,goal=r['goal'])
  messages=render_policy(policy);messages[1]['content'].reverse()
  ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
  lengths.append(len(ids)+visual-1)
 n=max(lengths);maxlen=max(maxlen,n)
 if n>10000:excluded.append(dict(id=r['id'],prefix_tokens=n))
 else:keep.append(r)
 if index%10000==0:print('COUNTED',index,flush=True)
allowed={r['id'] for r in keep};dest=W/'train-manifest.jsonl';dest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in keep))
counts=[]
for i in (1,2):
 p=W/f'phase{i}.jsonl';rows=[json.loads(l) for l in p.open()];rows=[r for r in rows if r['id'] in allowed]
 for j,r in enumerate(rows):r['draw']=j
 p.write_text(''.join(json.dumps(r)+'\n' for r in rows));counts.append(len(rows))
total=sum(counts)//256;warm=max(1,round(total*.01));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=W/'training-plan.json';plan=json.loads(p.read_text());plan.update(optimizer_steps=total,warmup_steps=warm,original_draws=sum(counts),used_draws=total*256,dropped_final_incomplete_batch=sum(counts)%256,samples=len(keep),manifest=str(dest),manifest_sha256=sha(dest),phase_sha256=[sha(W/f'phase{i}.jsonl') for i in (1,2)])
p.write_text(json.dumps(plan,indent=2))
p=W/'training_backend.py';p.write_text(p.read_text().replace('draws[:185600]',f'draws[:{total*256}]'))
p=W/'native.toml';p.write_text(p.read_text().replace('max_iter = 725',f'max_iter = {total}').replace('cycle_lengths = [725]',f'cycle_lengths = [{total}]').replace('warm_up_steps = [7]',f'warm_up_steps = [{warm}]'))
p=W/'config.json';cfg=json.loads(p.read_text());cfg['train']['max_iter']=total
from collections import Counter
nums=Counter(r['source'] for r in keep)
cfg['sources']={s:dict(count=n,weight=n/len(keep)) for s,n in nums.items()};p.write_text(json.dumps(cfg,indent=2))
(W/'token-admission.json').write_text(json.dumps(dict(retained=len(keep),excluded=excluded,max_before=maxlen,optimizer_steps=total),indent=2));print('ADMISSION_COMPLETE',len(keep),len(excluded),total,flush=True)
