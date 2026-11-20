import pathlib,json,collections
P=pathlib.Path('/rscratch/wzzheng/experiments/mixed-sft-preparation-20261028-v1')
actions={r['id']:r for r in map(json.loads,open(P/'artifacts/action-pair-audit.jsonl'))}
counts=collections.Counter();examples={}
def classify(r,a):
 n=r['current_original_step'];k=n['action'];info=n['info'];kind=a['action_type']
 if k=='TEXT':return 'exact_text_match' if kind=='Write' and a['value']==info else 'text_or_type_changed'
 if k=='COMPLETE':return 'terminal_type_match' if kind=='Terminate' else 'terminal_rewritten'
 if k=='CLICK' and isinstance(info,str):
  mapping={'KEY_HOME':'NavigateHome','KEY_BACK':'NavigateBack','KEY_ENTER':'KeyboardPress'}
  if info not in mapping:return 'unmapped_key'
  return 'key_type_match' if kind==mapping[info] else 'key_type_changed'
 if k in ('CLICK','LONG_PRESS'):
  expected='Click' if k=='CLICK' else 'LongPress'
  if kind!=expected:return 'action_type_changed'
  return 'coordinates_require_original_resolution_audit'
 if k=='SCROLL':return 'gesture_direction_distance_require_original_audit'
 return 'unmapped_action'
with (P/'artifacts/odyssey-action-diagnostic.jsonl').open('w') as out:
 for r in map(json.loads,open(P/'artifacts/odyssey-train-original-evidence-candidates.jsonl')):
  a=actions[r['id']]['action'];status=classify(r,a);counts[status]+=1
  obj=dict(id=r['id'],status=status,original=r['current_original_step']['action'],target=a['action_type'],fm_approved=False)
  out.write(json.dumps(obj)+'\n')
  examples.setdefault(status,dict(obj,original_info=r['current_original_step']['info'],target_action=a))
report=dict(counts=counts,examples=examples,training_ready=False,note='Exact text/type diagnostic only. Original screenshot resolution must be verified before coordinate conversion; no next-frame approval.')
(P/'artifacts/odyssey-action-diagnostic-report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(counts))
