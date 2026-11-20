
from pathlib import Path
import sys,json,math,time,hashlib,subprocess
W=Path('/data/mai/training/gui-libra-official-sft-20261117')
deadline=time.time()+600
while not (W/'excluded-official-format.json').exists():
 assert time.time()<deadline;time.sleep(3)
sys.path.insert(0,str(W))
from training_backend import validate_inputs
from native_protocol import render_policy,parse,HIGH_FIELDS
from mixed_native_targets import tokenize_targets
from transformers import AutoProcessor
from PIL import Image
import torch
torch.set_num_threads(4)
cfg,records,draws=validate_inputs(W/'config.json',W/'train-manifest.jsonl',[W/'phase1.jsonl',W/'phase2.jsonl'])
p=AutoProcessor.from_pretrained('/data/mai/benchmarks/gui-libra-base/ready-model',local_files_only=True)
examples={};longest=None;maxn=0;maxt=0
for num,r in enumerate(records.values()):
 with Image.open(r['current_image']) as im:w,h=im.size
 for level,v in r['views'].items():
  assert parse(v['official_target'])['fields']==v['action']
  policy=dict(current_image=None,official_system=v['official_system'],official_user=v['official_user'])
  msgs=render_policy(policy);msgs[1]['content'].reverse()
  n=len(p.tokenizer.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True))+round(w/32)*round(h/32)-1
  ids,weights=tokenize_targets(p.tokenizer,[v['official_target']],True);nt=len(ids[0]);maxt=max(maxt,nt)
  ex=dict(id=r['id'],source=r['source'],level=level,prefix_tokens=n,target_tokens=nt)
  examples.setdefault((r['source'],level),ex)
  if n>maxn:maxn=n;longest=ex
 if num%10000==0:print('VALIDATED',num,maxn,maxt,flush=True)
checks=list(examples.values())
if longest not in checks:checks.append(longest)
for ex in checks:
 r=records[ex['id']];v=r['views'][ex['level']]
 with Image.open(r['current_image']) as im:im=im.convert('RGB')
 msgs=render_policy(dict(current_image=im,official_system=v['official_system'],official_user=v['official_user']));msgs[1]['content'].reverse()
 inp=p(text=[p.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)],images=[im],return_tensors='pt',images_kwargs={'size':{'shortest_edge':1024,'longest_edge':max(16777216,math.ceil(im.height/32)*32*math.ceil(im.width/32)*32)}})
 assert inp['input_ids'].shape[-1]==ex['prefix_tokens']
 decoded=p.tokenizer.decode(inp['input_ids'][0]);assert v['official_user'] in decoded and v['official_system'] in decoded
 assert decoded.count('<|vision_start|>')==1 and decoded.rstrip().endswith('<|im_start|>assistant')
 assert '<thinking>' not in decoded and '<think>' not in decoded
 print('PROCESSOR_VERIFIED',ex['source'],ex['level'],ex['prefix_tokens'],flush=True)
report=dict(passed=True,records=len(records),draws=len(draws),max_prefix=maxn,max_target=maxt,checks=checks,fields=list(HIGH_FIELDS),all_original_targets_equal_parsed_labels=True,no_truncation=True)
(W/'processor-validation.json').write_text(json.dumps(report,indent=2))
(W/'token-admission.json').write_text(json.dumps(dict(retained=len(records),max_prefix=maxn,max_target=maxt,length_filter=None),indent=2))
print('VALIDATION_PASSED',len(records),maxn,maxt,flush=True)
plan=json.loads((W/'training-plan.json').read_text());plan.update(ready_to_launch=True,pending_checks=[]);(W/'training-plan.json').write_text(json.dumps(plan,indent=2))
proc=subprocess.Popen(['/data/mai/envs/mai/bin/python','-u',str(W/'launch.py')],stdin=subprocess.DEVNULL,stdout=(W/'run/launcher.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
print('LAUNCHER',proc.pid,flush=True)
