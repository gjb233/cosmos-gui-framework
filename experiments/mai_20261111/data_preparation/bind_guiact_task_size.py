import pathlib,json,collections,re,hashlib
from PIL import Image
P=pathlib.Path(__file__).resolve().parent
inv={r['id']:r for r in map(json.loads,(P/'artifacts/inventory.jsonl').open()) if r['source'].startswith('guiact')}
counts=collections.defaultdict(collections.Counter);out=[]
for r in map(json.loads,(P/'artifacts/original-trajectory-candidates.jsonl').open()):
 if not r['source'].startswith('guiact'):continue
 row=inv[r['id']];human='\n'.join(x['value'] for x in row['prefix'] if x['from']=='human')
 m=re.search(r'Instruction:\s*(.*?)(?:\n\n(?:Interaction History|Previous actions):|$)',human,re.S)
 goal=m[1].strip() if m else None
 with Image.open(r['current_image']) as im:size=im.size
 hits=[(i,n) for i,n in zip(r.get('row_indices',[]),r.get('original_records',[])) if n['question'].strip()==goal and (n['image_size']['width'],n['image_size']['height'])==size]
 hits=list({json.dumps(n,sort_keys=True,ensure_ascii=False):(i,n) for i,n in hits}.values())
 status='unique_uid_task_size_match' if len(hits)==1 else 'missing_original_train' if not r.get('original_records') else 'task_size_mismatch_or_conflict'
 counts[r['source']][status]+=1
 o=dict(id=r['id'],source=r['source'],status=status,goal=goal,image_size=size,old_join_status=r['status'],approved_for_training=False,fm_approved=False)
 if len(hits)==1:o.update(original_row_index=hits[0][0],original_record=hits[0][1],annotation_path=r['annotation_path'],annotation_sha256=r['annotation_sha256'])
 out.append(o)
with (P/'artifacts/guiact-task-size-bindings.jsonl').open('w') as f:
 for r in out:f.write(json.dumps(r,ensure_ascii=False)+'\n')
report=dict(counts=counts,training_ready=False)
(P/'artifacts/guiact-task-size-binding-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
