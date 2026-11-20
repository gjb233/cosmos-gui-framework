import json,pathlib,collections,re
P=pathlib.Path(__file__).resolve().parent
idx=collections.defaultdict(list)
for f in (P/'originals/mind2web-train').glob('*.index.jsonl'):
 for x in map(json.loads,f.open()):idx[(x['episode_id'],x['action_uid'])].append(x)
targets={x['id']:x for x in map(json.loads,(P/'artifacts/action-pair-audit.jsonl').open())}
counts=collections.Counter();ops=collections.Counter();examples={}
with (P/'artifacts/mind2web-original-action-candidates.jsonl').open('w') as out:
 for r in map(json.loads,(P/'artifacts/inventory.jsonl').open()):
  if 'mind2web' not in r['source'].lower():continue
  name=pathlib.Path(r['image']).stem
  hits=idx.get((name[:36],name[37:]),[])
  row=dict(id=r['id'],image=r['image'],source=r['source'],matches=hits,approved_for_training=False,fm_approved=False)
  if len(hits)!=1:status='missing_original_train' if not hits else 'ambiguous_original'
  else:
   raw=hits[0];row['official_train_identifier_verified']=True
   t=targets[r['id']];a=t.get('action',{});op=raw['operation'];ops[op['op']]+=1
   expected={'CLICK':'Click','TYPE':'Write','SELECT':'Select'}.get(op['op'])
   if not t.get('parsed'):status='target_parse_error'
   elif a.get('action_type')!=expected:status='action_type_changed'
   elif op['op'] in ('TYPE','SELECT'):
    status='exact_type_and_text' if a.get('value')==op['value'] else 'text_changed'
   else:status='click_target_identity_pending'
   row['published_action']=a
  row['status']=status;counts[status]+=1
  examples.setdefault(status,dict(id=r['id'],image=r['image']))
  out.write(json.dumps(row,ensure_ascii=False)+'\n')
report=dict(rows=sum(counts.values()),counts=counts,original_operations=ops,examples=examples,training_ready=False,note='Official train episode/action binding only; original image identity, target geometry, Low semantics and next frame still require verification.')
(P/'artifacts/mind2web-original-action-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
