"""Join every AMEX row to pinned official instruction/action chains; diagnose, not approve."""
import pathlib,json,zipfile,re,hashlib,collections
P=pathlib.Path(__file__).resolve().parent
targets={r['id']:r for r in map(json.loads,(P/'artifacts/action-pair-audit.jsonl').open()) if r['source']=='amex'}
index=collections.defaultdict(list);origtypes=collections.Counter()
with zipfile.ZipFile(P/'originals/amex-instruction-anno.zip') as z:
 for name in z.namelist():
  if not name.endswith('.json'):continue
  b=z.read(name);r=json.loads(b);ep=r['episode_id'];steps=r['steps']
  for i,n in enumerate(steps):
   origtypes[n['action']]+=1
   index[(ep,n['step_id'])].append(dict(annotation_entry=name,annotation_sha256=hashlib.sha256(b).hexdigest(),goal=r['instruction'],original_step=n,
     next_original_step=steps[i+1] if i+1<len(steps) else None,previous_original_steps=steps[max(0,i-4):i]))
counts=collections.Counter();examples={}
with (P/'artifacts/amex-original-action-candidates.jsonl').open('w') as f:
 for r in map(json.loads,(P/'artifacts/inventory.jsonl').open()):
  if r['source']!='amex':continue
  name=pathlib.Path(r['image']).name
  m=re.fullmatch(r'([0-9a-f]{32})step(\d+)\.png',name)
  out=dict(id=r['id'],source='amex',image=r['image'],approved_for_training=False,fm_approved=False)
  if not m:out['status']='unparsed_image_id'
  else:
   ep,step=m[1],int(m[2]);hits=index.get((ep,step+1),[])
   out.update(episode_id=ep,step=step,original_step_id=step+1,matches=hits)
   if len(hits)!=1:out['status']='missing_original' if not hits else 'ambiguous_original'
   elif not targets[r['id']].get('parsed'):out['status']='published_target_parse_error'
   else:
    n=hits[0]['original_step'];a=targets[r['id']]['action'];k=n['action'];typ=a['action_type']
    if k=='TYPE':status='exact_text' if typ=='Write' and a['value']==n['type_text'] else 'text_or_type_changed'
    elif k in ('TAP','LONG_PRESS'):
     expected={'TAP':'Click','LONG_PRESS':'LongPress'}[k]
     if typ!=expected:status='type_changed'
     elif not isinstance(a['point_2d'],list):status='point_missing'
     else:
      pt=[round(n['touch_coord'][i]/n['device_dim'][i]*1000) for i in range(2)]
      out['normalized_original_point']=pt;out['point_delta']=max(abs(pt[i]-a['point_2d'][i]) for i in range(2))
      status='exact_rounded_point' if pt==a['point_2d'] else 'point_changed'
    elif k=='SWIPE':status='swipe_direction_and_extent_pending'
    elif k in ('PRESS_ENTER','PRESS_HOME','PRESS_BACK'):
     expected={'PRESS_ENTER':'KeyboardPress','PRESS_HOME':'NavigateHome','PRESS_BACK':'NavigateBack'}[k]
     status='key_type_match_parameters_pending' if typ==expected else 'key_type_changed'
    elif k in ('TASK_COMPLETE','TASK_IMPOSSIBLE'):
     expected_value='success' if k=='TASK_COMPLETE' else 'failed'
     status='terminal_type_and_value_match' if typ=='Terminate' and a['value']==expected_value else 'terminal_rewritten'
    else:status='original_action_mapping_pending'
    out['status']=status;out['published_action']=a
    nxt=hits[0]['next_original_step'];out['next_step_consecutive']=bool(nxt and nxt['step_id']==step+2)
  counts[out['status']]+=1;examples.setdefault(out['status'],out);f.write(json.dumps(out,ensure_ascii=False)+'\n')
report=dict(rows=sum(counts.values()),published_step_base=0,original_step_base=1,counts=counts,original_action_types=origtypes,examples={k:dict(id=v['id'],image=v['image']) for k,v in examples.items()},training_ready=False,
 note='Pinned official AMEX annotation join and action diagnostic; split/image identity/semantic Low approval still required')
(P/'artifacts/amex-original-action-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
