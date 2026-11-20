from pathlib import Path
import os,sys,json,math,statistics,collections,itertools,concurrent.futures,shutil
from PIL import Image
from transformers import AutoProcessor
root=Path(__file__).resolve().parent
src=root.parent/'mai-ui-approved70933-1f1b-micro1-gb64-fresh-a21-gpu17-save50/1f1b'
sys.path.insert(0,str(src))
from native_protocol import render_policy,target
proc=AutoProcessor.from_pretrained('/rscratch/wzzheng/experiments/gui-mai-lowres-abd-a23-20261005-v1/model',local_files_only=True)
tok=proc.tokenizer
rows=list(map(json.loads,(src/'artifacts/train-manifest.jsonl').open()))
assert len(rows)==len({r['id'] for r in rows})==70933
def dims(path):
 with Image.open(path) as im:return im.size
paths=list(dict.fromkeys(r['current_image'] for r in rows))
with concurrent.futures.ThreadPoolExecutor(16) as ex:sizes=dict(zip(paths,ex.map(dims,paths)))
def msgs(r,l):
 return render_policy(dict(current_image=r['current_image'],instruction=r['views'][l]['instruction'],history=r['history_mobile_use'],level=l,goal=r['goal']))
sample=msgs(rows[0],next(iter(rows[0]['views'])));sample[1]['content'][0]['text']='UNIQUE_PLACEHOLDER'
template=proc.apply_chat_template(sample,tokenize=False,add_generation_prompt=True,enable_thinking=False)
assert template.count('UNIQUE_PLACEHOLDER')==template.count('<|image_pad|>')==1
lens={};dist=[]
with (root/'lengths.jsonl').open('x') as log:
 for start in range(0,len(rows),64):
  batch=[];texts=[];answers=[]
  for r in rows[start:start+64]:
   w,h=sizes[r['current_image']]
   visual=max(1,round(w/32))*max(1,round(h/32))
   for l,v in r['views'].items():
    m=msgs(r,l);prompt=template.replace('UNIQUE_PLACEHOLDER',m[1]['content'][0]['text'])
    texts.append(prompt);answers.append(target(v['native_action']));batch.append((r,l,visual))
  lengths=tok(texts,add_special_tokens=True,return_length=True,verbose=False)['length'];targets=tok(answers,add_special_tokens=False,return_length=True,verbose=False)['length']
  for (r,l,vis),n,ans in zip(batch,lengths,targets):
   length=n-1+vis+ans+1
   if start==0 and l==next(iter(r['views'])) and r is rows[0]:
    with Image.open(r['current_image']) as im:
     w,h=im.size
     inp=proc(text=[proc.apply_chat_template(msgs(r,l),tokenize=False,add_generation_prompt=True,enable_thinking=False)],images=[im.convert('RGB')],return_tensors='pt',images_kwargs={'size':{'shortest_edge':1024,'longest_edge':max(16777216,math.ceil(w/32)*32*math.ceil(h/32)*32)}})
     assert inp.input_ids.shape[-1]+ans+1==length,(inp.input_ids.shape,length,vis)
   lens.setdefault(r['id'],{})[l]=length
   log.write(json.dumps(dict(id=r['id'],level=l,source=r['source'],visual=vis,text=n-1,target=ans+1,total=length))+'\n')
  if start%4096==0:print('count',start,flush=True)
phases=[list(map(json.loads,(src/f'artifacts/schedule-phase{i}.jsonl').open())) for i in (1,2)]
risk={i:lens[d['id']][d['level']] for i,d in enumerate(phases[0]) if i in (1072,1073,1074,1075,1456,1457)}
cut=min(8192,2**int(math.floor(math.log2(max(risk[1456],risk[1457])-1))))
rowmax={k:max(v.values()) for k,v in lens.items()};keep={k for k,v in rowmax.items() if v<=cut}
out=root/'prepared';out.mkdir()
def write(name,data):
 with (out/name).open('x') as f:
  for r in data:f.write(json.dumps(r,ensure_ascii=False)+'\n')
write('train-manifest.jsonl',(r for r in rows if r['id'] in keep));write('quarantine.jsonl',(r for r in rows if r['id'] not in keep))
counts=[]
for i,ph in enumerate(phases,1):
 valid=[dict(d,draw=j) for j,d in enumerate(d for d in ph if d['id'] in keep)]
 write(f'schedule-phase{i}.jsonl',valid);counts.append(len(valid))
def stats(v):
 v=sorted(v);return dict(n=len(v),p50=v[math.ceil(.50*len(v))-1],p90=v[math.ceil(.9*len(v))-1],p95=v[math.ceil(.95*len(v))-1],p99=v[math.ceil(.99*len(v))-1],max=v[-1],over={str(c):sum(x>c for x in v) for c in (4096,8192,16384,32768,65536)})
report=dict(threshold=cut,rows=stats(rowmax.values()),draws=stats(lens[d['id']][d['level']] for ph in phases for d in ph),risk=risk,kept=len(keep),quarantined=len(rows)-len(keep),retained_phases=counts,excluded_sources=dict(collections.Counter(r['source'] for r in rows if r['id'] not in keep)))
assert sum(counts)<=212912 and len(keep)>0
(root/'report.tmp').write_text(json.dumps(report,indent=2));(root/'report.tmp').replace(root/'report.json');print(report,flush=True)
