"""Incrementally join only SHA-verified downloaded Odyssey train annotations."""
import json,pathlib,hashlib,collections,fcntl
P=pathlib.Path('/rscratch/wzzheng/experiments/mixed-sft-preparation-20261028-v1')
out=P/'artifacts/odyssey-train-original-evidence-candidates.jsonl'
lock=(P/'odyssey-incremental-join.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
verified={}
for name in ('retry-full-tree-download-results.jsonl','backoff-download-results.jsonl'):
 for line in open(P/'originals/odyssey-train'/name):
  try:r=json.loads(line)
  except json.JSONDecodeError:continue
  if r['status']=='verified':verified[r['episode_file']]=r
done={r['id'] for r in map(json.loads,open(out))} if out.exists() else set()
cache={};count=0
with out.open('a') as f:
 for r in map(json.loads,open(P/'artifacts/inventory.jsonl')):
  if r['source']!='gui-odyssey' or r['id'] in done:continue
  ep,step=r['image'].removesuffix('.png').rsplit('_',1);step=int(step);meta=verified.get(ep+'.json')
  if not meta:continue
  if ep not in cache:
   raw=pathlib.Path(meta['path']).read_bytes()
   assert hashlib.sha256(raw).hexdigest()==meta['sha256']
   cache[ep]=json.loads(raw)
  data=cache[ep];hits=[x for x in data['steps'] if x['step']==step and x['screenshot']==r['image']]
  if len(hits)!=1:continue
  current=hits[0];following=[x for x in data['steps'] if x['step']==step+1]
  result=dict(id=r['id'],source=r['source'],episode_id=ep,step=step,annotation_path=meta['path'],annotation_sha256=meta['sha256'],revision=meta['revision'],
   split='random_train',current_original_step=current,goal=data['task_info']['instruction'],
   low_instruction=current.get('low_level_instruction'),next_original_step=following[0] if len(following)==1 else None,
   previous_original_steps=[x for x in data['steps'] if step-4<=x['step']<step],
   current_image_identity_verified=False,action_alignment_verified=False,fm_approved=False,approved_for_training=False)
  f.write(json.dumps(result)+'\n');count+=1
allrows=list(map(json.loads,open(out)))
report=dict(joined_rows=len(allrows),added_rows=count,episodes=len({x['episode_id'] for x in allrows}),with_low=sum(bool(x['low_instruction']) for x in allrows),
 with_next_original_step=sum(x['next_original_step'] is not None for x in allrows),training_ready=False,fm_approved=0)
(P/'artifacts/odyssey-incremental-original-report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
