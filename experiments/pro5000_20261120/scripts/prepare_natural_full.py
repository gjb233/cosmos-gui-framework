
from pathlib import Path
import json,sys,hashlib,collections,math
from PIL import Image
from transformers import AutoTokenizer
O=Path('/data/mai/training/gui-libra-natural-history-20261117')
W=Path('/data/mai/training/gui-libra-natural-history-full-20261117')
B=Path('/data/mai/assets/training-ready')
S=Path('/data/mai/assets/training-ready-strict-10k-20261113')
W.mkdir(exist_ok=False);(W/'run/audit').mkdir(parents=True)
for p in O.iterdir():
 if p.suffix in ('.py','.toml') or p.name in ('config.json','training-plan.json','templates.json'):
  (W/p.name).write_text(p.read_text().replace(str(O),str(W)))
def rows(p):return [json.loads(l) for l in p.open()]
def write(p,r):p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in r))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
old={r['id']:r for r in rows(O/'train-manifest.jsonl')}
approved={r['id'] for r in rows(S/'train-manifest.jsonl')}
length_only={r['id'] for r in rows(S/'excluded-samples.jsonl') if r['reasons']==['input_prefix_over_10000']}
approved |= length_only
base=rows(B/'train-manifest.jsonl');selected=[]
for r in base:
 if r['id'] not in approved:continue
 if r['id'] in old:r=old[r['id']]
 else:
  r['published_history']=list(r['history'])
  r['history_description_policy']='all provided original natural-language history; no synthetic completion'
 assert all(isinstance(x,str) for x in r['history'])
 selected.append(r)
assert len(selected)==62800
assert set(old)<=approved
write(W/'train-manifest.jsonl',selected)
counts=[]
for i in (1,2):
 phase=[d for d in rows(B/f'schedule-phase{i}.jsonl') if d['id'] in approved]
 for j,d in enumerate(phase):d.update(draw=j,variant='direct')
 write(W/f'phase{i}.jsonl',phase);counts.append(len(phase))
steps=sum(counts)//256;warm=max(1,round(steps*.01))
plan=json.loads((W/'training-plan.json').read_text())
plan.update(optimizer_steps=steps,warmup_steps=warm,original_draws=sum(counts),used_draws=steps*256,dropped_final_incomplete_batch=sum(counts)%256,samples=len(selected),manifest=str(W/'train-manifest.jsonl'),manifest_sha256=sha(W/'train-manifest.jsonl'),phase_sha256=[sha(W/f'phase{i}.jsonl') for i in (1,2)],max_input_tokens=None,ready_to_launch=False,pending_checks=['full_length_processor_validation'],restored_length_filtered_samples=len(approved-set(old)))
(W/'training-plan.json').write_text(json.dumps(plan,indent=2))
cfg=json.loads((W/'config.json').read_text());cfg['train']['max_iter']=steps
nums=collections.Counter(r['source'] for r in selected)
cfg['sources']={s:dict(count=n,weight=n/len(selected)) for s,n in nums.items()}
for i,n in enumerate(counts,1):cfg['sampling'][f'phase{i}']['draws']=n
cfg['max_input_tokens']=None;cfg['truncate_inputs']=False
(W/'config.json').write_text(json.dumps(cfg,indent=2))
p=W/'training_backend.py';s=p.read_text()
line="  assert inp['input_ids'].shape[-1]<=10000, ('input token limit',inp['input_ids'].shape[-1])"
assert line in s;s=s.replace(line,"  # Keep full input; admission does not impose a token length cutoff.")
s=s.replace('draws[:185600]',f'draws[:{steps*256}]');p.write_text(s)
p=W/'native.toml';s=p.read_text().replace('max_iter = 725',f'max_iter = {steps}').replace('cycle_lengths = [725]',f'cycle_lengths = [{steps}]').replace('warm_up_steps = [7]',f'warm_up_steps = [{warm}]');p.write_text(s)
sys.path.insert(0,str(W))
from native_protocol import target,render_policy,parse
tok=AutoTokenizer.from_pretrained('/data/mai/benchmarks/gui-libra-base/ready-model',local_files_only=True)
examples={};lengths=[];longest=None;maximum=0;max_target=0
for num,r in enumerate(selected):
 with Image.open(r['current_image']) as im:w,h=im.size
 for level,v in r['views'].items():
  messages=render_policy(dict(current_image=None,image_size=(w,h),instruction=v['instruction'],history=r['history'],level=level,goal=r['goal']))
  messages[1]['content'].reverse()
  n=len(tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True))+round(w/32)*round(h/32)-1
  answer=target(v['action'],level);parse(answer)
  nt=len(tok(answer,add_special_tokens=False)['input_ids'])+1
  ex=dict(id=r['id'],source=r['source'],level=level,prefix_tokens=n,target_tokens=nt)
  examples.setdefault((r['source'],level),ex)
  if n>maximum:maximum=n;longest=ex
  max_target=max(max_target,nt);lengths.append(ex)
 if num%10000==0:print('COUNTED',num,maximum,flush=True)
audit=list(examples.values())
if longest not in audit:audit.append(longest)
(W/'prompt-examples.json').write_text(json.dumps(audit,indent=2))
write(W/'input-lengths.jsonl',lengths)
report=dict(retained=len(selected),restored=len(approved-set(old)),original_length_exclusions=len(length_only),secondary_length_exclusions=len(approved-set(old))-len(length_only),excluded_uncertain=7483,max_prefix=maximum,max_target=max_target,over_10000=sum(x['prefix_tokens']>10000 for x in lengths),length_filter=None,truncation=False,optimizer_steps=steps,source_counts=dict(nums),template_sha256=sha(W/'templates.json'))
(W/'token-admission.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
