import collections, concurrent.futures, hashlib, json, pathlib, re, fcntl
P=pathlib.Path('/rscratch/wzzheng/experiments/mixed-sft-preparation-20261028-v1')
R=pathlib.Path('/rscratch/wzzheng/experiments/gui-libra-low-terminal-cefm500-a22-20261014-v1/train.jsonl')
O=pathlib.Path('/rscratch/wzzheng/experiments/guiodyssey-high-a27-20261028-v1/data')
held=json.load(open(P/'heldout.json'))
excluded={tuple(x) for x in held['episodes']}
images={r['id']:r for r in map(json.loads,open(P/'artifacts/image-decode-audit.jsonl'))}
ac={(r['episode_id'],r['step']):r for r in map(json.loads,open(R))}
split=json.load(open(O/'splits/random_split.json'))
train=set(split['train'])
def check(r):
 result=dict(id=r['id'],source=r['source'],approved_for_training=False)
 try:
  if r['source']=='android_control':
   match=re.fullmatch(r'(\d+)/screenshot_(\d+)\.png',r['image']);assert match
   episode,step=match[1],int(match[2]); native=ac.get((episode,step))
   if native is None:return dict(result,status='missing_from_existing_train_manifest')
   result.update(annotation_path=str(R),low_instruction=native['instruction'],goal=native['goal'],native_action=native['action'],next_image=native['next_image'])
   result['current_image_sha_matches']=hashlib.sha256(pathlib.Path(native['current_image']).read_bytes()).hexdigest()==images[r['id']]['sha256']
   result['split_train']=native['split']=='train'
  else:
   match=re.fullmatch(r'(\d+)_(\d+)\.png',r['image']);assert match
   episode,step=match[1],int(match[2])
   candidates=[O/'annotations'/(episode+'.json'),pathlib.Path('/rscratch/wzzheng/data/mai/GUIOdyssey/annotations')/(episode+'.json')]
   path=next((p for p in candidates if p.is_file()),None)
   if path is None:return dict(result,status='original_annotation_missing')
   data=json.load(open(path)); hits=[x for x in data['steps'] if x['step']==step and x['screenshot']==r['image']]
   assert len(hits)==1;native=hits[0]
   result.update(annotation_path=str(path),low_instruction=native.get('low_level_instruction'),goal=data['task_info']['instruction'],native_action={'action':native['action'],'info':native['info']})
   result['split_train']=episode+'.json' in train
   following=[x for x in data['steps'] if x['step']==step+1]
   result['next_original_step']=following[0] if len(following)==1 else None
  result.update(episode_id=episode,step=step,heldout_episode=(r['source'],episode) in excluded,status='candidate_requires_action_alignment')
 except Exception as e:result.update(status='error',error=repr(e))
 return result
with (P/'native-low-join.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 records=[r for r in map(json.loads,open(P/'artifacts/inventory.jsonl')) if r['source'] in ('android_control','gui-odyssey')]
 counts=collections.defaultdict(collections.Counter)
 with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool,(P/'artifacts/native-low-candidates.jsonl').open('w') as out:
  for i,r in enumerate(pool.map(check,records)):
   out.write(json.dumps(r,ensure_ascii=False)+'\n');counts[r['source']][r['status']]+=1
   if r.get('low_instruction'):counts[r['source']]['has_low_text']+=1
   if r.get('heldout_episode'):counts[r['source']]['heldout_episode']+=1
   if i%1000==0:out.flush();print(i,dict(counts),flush=True)
 report=dict(sources=dict(counts),training_ready=False)
 (P/'artifacts/native-low-candidate-report.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report),flush=True)
