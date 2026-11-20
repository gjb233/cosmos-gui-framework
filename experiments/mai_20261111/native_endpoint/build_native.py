from pathlib import Path
import json,collections,sys,hashlib,ast
N=Path(__file__).resolve().parent;P=N.parent;sys.path.insert(0,str(N))
from native_protocol import published,target,parse,ac_text,BASE_SYSTEM,SYSTEM
S=P/'mai-ui-allsource-fm640-a18-20261104'
rows=[json.loads(l) for l in (S/'artifacts/train-manifest.jsonl').open()]
byid={r['id']:r for r in rows};bindings={}
spec=['aitw-official-incremental-bindings.jsonl','amex-original-action-candidates.jsonl','original-trajectory-candidates.jsonl','mind2web-original-action-candidates.jsonl','odyssey-train-original-evidence-candidates.jsonl','ac-original-action-audit.jsonl']
for fn in spec:
 for l in (P/'artifacts'/fn).open():
  e=json.loads(l);r=byid.get(e['id'])
  if r is None:continue
  src=r['source'];ep=step=None
  if src=='aitw' and e.get('status')=='unique_original_record':
   ep=e['original_episode_id'];step=e['step']
  elif src=='amex' and len(e.get('matches',[]))==1:
   ep=e['episode_id'];step=e['step']
  elif src in ['coat-terminal','guiact-web','guiact-web-chinese'] and e.get('status')=='unique_original_content':
   ep=e['episode_id'];step=e['step']
  elif src=='mind2web' and len(e.get('matches',[]))==1 and e['matches'][0]['split']=='train':
   ep=e['matches'][0]['episode_id'];step=e['matches'][0]['step']
  elif src=='gui-odyssey':
   ep=e['episode_id'];step=e['step']
  elif src=='android_control' and e.get('official_train') and not e.get('heldout_episode'):
   ep=e['episode_id'];step=e['step']
  if ep is not None:bindings[r['id']]=(src,str(ep),int(step))
print('bindings',len(bindings),flush=True)
groups=collections.defaultdict(lambda:collections.defaultdict(list))
actions=collections.Counter()
for r in rows:
 for v in r['views'].values():
  v['native_action']=published(v['action'])
  v['native_target']=target(v['native_action']);assert parse(v['native_target'])==v['native_action']
 a=next(iter(r['views'].values()))['native_action'];actions[a['arguments']['action']]+=1
 b=bindings.get(r['id'])
 if b:groups[b[:2]][b[2]].append(a)
def ac_action(a,w,h):
 t=a.get('action_type',a.get('type'));d={}
 if t in ['click','long_press']:
  d={'action':t,'coordinate':[min(999,max(0,round(a['x']/w*999))),min(999,max(0,round(a['y']/h*999)))]}
 elif t=='input_text':d={'action':'type','text':a['text']}
 elif t=='scroll':d={'action':'swipe','direction':a['direction']}
 elif t in ['navigate_home','navigate_back']:d={'action':'system_button','button':t.removeprefix('navigate_')}
 elif t=='open_app':d={'action':'open','text':a['app_name']}
 elif t=='wait':d={'action':'wait'}
 elif t=='keyboard_enter':d={'action':'system_button','button':'enter'}
 elif t=='status':d={'action':'terminate','status':a.get('goal_status','success')}
 else:raise ValueError(a)
 return {'name':'mobile_use','arguments':d}
counts=collections.defaultdict(collections.Counter);examples={}
out=N/'artifacts/train-manifest.jsonl'
with out.open('w') as f:
 for r in rows:
  src=r['source'];b=bindings.get(r['id']);hist=[];steps=[];policy='no verified trajectory binding; empty history'
  full=False
  if b:
   _,ep,step=b
   # Retained published training actions only; never infer coordinates from descriptions.
   for st,aa in sorted(groups[b[:2]].items()):
    if st>=step:continue
    unique={json.dumps(a,sort_keys=True):a for a in aa}
    if len(unique)==1:hist.append(next(iter(unique.values())));steps.append(st)
   policy='all available prior retained published training actions at verified original trajectory indices'
   full=steps==list(range(step))
   if src=='android_control':
    p=P/'originals/androidcontrol-selected'/f'{ep}.json'
    if p.exists():
     raw=json.loads(p.read_text())
     if step<=len(raw['actions']):
      hist=[ac_action(a,raw['screenshot_widths'][i],raw['screenshot_heights'][i]) for i,a in enumerate(raw['actions'][:step])]
      steps=list(range(step));full=True;policy='complete official AndroidControl prior actions; per-frame dimensions'
   assert all(st<step for st in steps)
  for a in hist:parse(target(a))
  r['history_mobile_use']=hist
  r['history_native_audit']={'binding':b,'original_steps':steps,'complete_from_start':full,'policy':policy,'truncated':False}
  counts[src]['records']+=1;counts[src]['fm']+=r['fm_mask'];counts[src]['history_actions']+=len(hist);counts[src]['complete_history']+=int(full);counts[src]['unbound']+=int(b is None)
  examples.setdefault(src,{'id':r['id'],'text':ac_text({'goal':r['goal'],'level':'high','step_instruction':'','history_mobile_use':hist}),'target':next(iter(r['views'].values()))['native_target'],'history_audit':r['history_native_audit']})
  f.write(json.dumps(r,ensure_ascii=False)+'\n')
# Exact user-text equality with evaluation renderer, AST extracted to avoid GPU deps.
evalpath=P/'evaluation-mai-base-ac-user-template-a18-20261104/grounding/native.py'
tree=ast.parse(evalpath.read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='ac_text')
ns={'json':json};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(evalpath),'exec'),ns)
vr=[json.loads(l) for l in (evalpath.parent/'manifest.jsonl').open()]
assert all(ac_text(r)==ns['ac_text'](r) for r in vr)
# Assert original labels, inputs and FM transfers unchanged.
old=[json.loads(l) for l in (S/'artifacts/train-manifest.jsonl').open()]
for a,b in zip(old,rows):
 for k in ['id','source','goal','current_image','current_image_sha256','fm_mask','next_image','history']:
  assert a[k]==b[k],(a['id'],k)
 for lev in a['views']:assert a['views'][lev]['action']==b['views'][lev]['action']
report={'records':len(rows),'sources':8,'fm':sum(r['fm_mask'] for r in rows),'counts':counts,'native_actions':actions,'eval_user_text_exact_match':len(vr),'system_base_sha256':hashlib.sha256(BASE_SYSTEM.encode()).hexdigest(),'extended_system_sha256':hashlib.sha256(SYSTEM.encode()).hexdigest(),'action_extensions_user_authorized':True,'fresh_base':True,'original_label_preservation':True,'history_policy':'AC full official; other sources all available verified prior retained training actions, coverage recorded; not a claim of complete official history for all datasets','coordinate_mapping':'published normalized 0..1000 to 0..999; AC histories per-frame pixels to 999'}
(N/'protocol-audit.json').write_text(json.dumps(report,indent=2))
(N/'prompt-examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2))
(N/'data-ready.json').write_text(json.dumps({'passed':True,'records':len(rows),'fm':report['fm']}))
print(json.dumps(report),flush=True)
