
from pathlib import Path
import json,collections,re,sys,hashlib,ast
from PIL import Image
O=Path('/data/mai/training/gui-libra-natural-history-full-20261117')
W=Path('/data/mai/training/gui-libra-official-sft-20261117')
assert not (W/'run/active.json').exists();W.mkdir(exist_ok=True);(W/'run/audit').mkdir(parents=True,exist_ok=True)
for p in O.iterdir():
 if p.is_file() and (p.suffix in ('.py','.toml') or p.name in ('config.json','training-plan.json','phase1.jsonl','phase2.jsonl')):
  (W/p.name).write_text(p.read_text().replace(str(O),str(W)))
rows=[json.loads(l) for l in (O/'train-manifest.jsonl').open()]
bad='7bfdcd0933289baa155942782bf682c8d4b3aa0d2c2f4184be4c167376678cfe:0'
rows=[r for r in rows if r['id']!=bad]
counts=[]
for i in (1,2):
 p=W/f'phase{i}.jsonl';ds=[json.loads(l) for l in p.open()];ds=[d for d in ds if d['id']!=bad]
 for j,d in enumerate(ds):d['draw']=j
 p.write_text(''.join(json.dumps(d)+'\n' for d in ds));counts.append(len(ds))
root=Path('/data/mai/assets/gui-libra-81k-sft-20261028-v1/snapshot')
systems=set();errors=[];derived=0;exact=0
for source in sorted({r['source'] for r in rows}):
 selected=[r for r in rows if r['source']==source]
 path=root/selected[0]['published_annotation_provenance']['direct']
 idx=collections.defaultdict(list)
 for num,d in enumerate(json.loads(path.read_text())):
  cv=d['conversations'];a=next(c['value'] for c in cv if c['from'] in ('gpt','assistant'))
  try:answer=json.loads(re.search(r'<answer>\s*(.*?)\s*</answer>',a,re.S)[1])
  except Exception:continue
  u=next(c['value'] for c in cv if c['from'] in ('human','user'))
  sy=next(c['value'] for c in cv if c['from']=='system')
  m=re.search(r'(Instruction: )(.*?)(\n\s*Interaction History:)',u,re.S)
  if not m:continue
  idx[(Path(d['image']).name,json.dumps(answer,sort_keys=True,ensure_ascii=False))].append(dict(system=sy,user=u,target=a,index=num,instruction=m[2]))
 for r in selected:
  for level,v in r['views'].items():
   hits=idx[(Path(r['current_image']).name,json.dumps(v['action'],sort_keys=True,ensure_ascii=False))]
   norm=lambda s:' '.join(s.split())
   desired=r['goal'] if level=='high' else v['instruction']
   same=[h for h in hits if norm(h['instruction'])==norm(desired)]
   if not same and level=='low':same=[h for h in hits if norm(h['instruction'])==norm(r['goal'])]
   unique={(h['system'],h['user'],h['target']):h for h in same}
   if len(unique)!=1:errors.append(dict(id=r['id'],source=source,level=level,matches=len(unique)));continue
   h=next(iter(unique.values()));user=h['user'];is_derived=norm(h['instruction'])!=norm(desired)
   if is_derived:
    m=re.search(r'(Instruction: )(.*?)(\n\s*Interaction History:)',user,re.S)
    user=user[:m.start(2)]+desired+user[m.end(2):];derived+=1
   else:exact+=1
   assert user.count('<image>')==1 and user.startswith('<image>\n')
   assert '<thinking>' not in h['target'] and '<think>' not in h['target']
   v.update(official_system=h['system'],official_user=user[len('<image>\n'):],official_target=h['target'],official_sft_provenance=dict(path=str(path),index=h['index'],derived_low_instruction=is_derived))
   systems.add(h['system'])
 print('MATCHED',source,'errors',len(errors),flush=True)
(W/'source-match-audit.json').write_text(json.dumps(dict(errors=errors,exact_views=exact,derived_low_views=derived,systems=len(systems)),indent=2))
assert not errors,errors[:5]
(W/'official-systems.json').write_text(json.dumps(sorted(systems),ensure_ascii=False,indent=2))
(W/'train-manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
protocol='''import json,re,math
from pathlib import Path
SYSTEMS=set(json.loads((Path(__file__).parent/'official-systems.json').read_text()))
SYSTEM=sorted(SYSTEMS)[0]
HIGH_FIELDS=LOW_FIELDS=('action_description','action_type','action_target','value','point_2d')
SPEC={}
def parse(text):
 m=re.fullmatch(r'<answer>\\s*(.*?)\\s*</answer>',text.strip(),re.S)
 assert m,text
 a=json.loads(m[1]);assert set(a)==set(HIGH_FIELDS)
 p=a['point_2d'];assert p is None or p=='None' or (isinstance(p,list) and len(p)==2 and all(isinstance(x,(int,float)) and math.isfinite(x) for x in p))
 return dict(action=a['action_type'],fields=a)
def target(a,level='low',reasoning=None):
 text='<answer>\\n'+json.dumps({k:a[k] for k in HIGH_FIELDS},ensure_ascii=False,indent=4)+'\\n</answer>'
 parse(text);return text
def render_policy(policy):
 assert policy['official_system'] in SYSTEMS
 return [{'role':'system','content':policy['official_system']},{'role':'user','content':[{'type':'text','text':policy['official_user']},{'type':'image','image':policy['current_image']}]}]
'''
(W/'native_protocol.py').write_text(protocol)
p=W/'mixed_native_sample.py';s=p.read_text().replace("history=record['history'],level=level,goal=record['goal'])","history=record['history'],level=level,goal=record['goal'],official_system=view['official_system'],official_user=view['official_user'])").replace("target(view['action'],level)","view['official_target']");p.write_text(s)
p=W/'training_backend.py';s=p.read_text().replace("history=c['history'],goal=c['goal'])","history=c['history'],goal=c['goal'],official_system=c['official_system'],official_user=c['official_user'])").replace("record_id=r['id'])","record_id=r['id'],official_system=r['views'][level]['official_system'],official_user=r['views'][level]['official_user'])");p.write_text(s)
p=W/'patch_training.py';s=p.read_text().replace("assert s==SPEC[level+'_system']","assert s==json.loads(c)['official_system']");p.write_text(s)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=W/'training-plan.json';plan=json.loads(p.read_text());plan.update(ready_to_launch=False,pending_checks=['official_sft_processor_validation'],manifest_sha256=sha(W/'train-manifest.jsonl'),protocol='Official GUI-Libra source conversations, direct-only five fields; Low instruction substitution when no published Low conversation',samples=len(rows),original_draws=sum(counts),used_draws=sum(counts)//256*256,dropped_final_incomplete_batch=sum(counts)%256,phase_sha256=[sha(W/f'phase{i}.jsonl') for i in (1,2)],parent_run=str(O),fresh_optimizer=True,fresh_lora=True,template_sha256=sha(W/'official-systems.json'));p.write_text(json.dumps(plan,indent=2))
p=W/'config.json';cfg=json.loads(p.read_text());cfg['protocol']=plan['protocol'];cfg['templates']=dict(training='verbatim source system and user, original assistant answer; derived Low only substitutes Instruction',fields=list(['action_description','action_type','action_target','value','point_2d']),thinking=False);p.write_text(json.dumps(cfg,indent=2))
for p in W.glob('*.py'):ast.parse(p.read_text())
print('PREPARED',len(rows),exact,derived,flush=True)

p=W/'config.json';cfg=json.loads(p.read_text());nums=collections.Counter(r['source'] for r in rows);cfg['sources']={k:dict(count=v,weight=v/len(rows)) for k,v in nums.items()}
for i,n in enumerate(counts,1):cfg['sampling'][f'phase{i}']['draws']=n
p.write_text(json.dumps(cfg,indent=2))
(W/'excluded-official-format.json').write_text(json.dumps(dict(id=bad,reason='malformed published assistant JSON; no manual target repair')))
