"""Compare every retained Odyssey screenshot to its original archive and action.
Does not approve altered actions, tolerate coordinate changes, or enable FM.
"""
import json,pathlib,collections,concurrent.futures,hashlib,io,zipfile,threading,fcntl
from PIL import Image
P=pathlib.Path(__file__).resolve().parent
Z=pathlib.Path('/rscratch/wzzheng/data/mai/GUIOdyssey/screenshots/screenshots-unsplit.zip')
rows=lambda p:map(json.loads,p.open())
targets={r['id']:r['action'] for r in rows(P/'artifacts/action-pair-audit.jsonl') if r.get('parsed') and r['source']=='gui-odyssey'}
images={r['id']:r for r in rows(P/'artifacts/image-decode-audit.jsonl') if r['source']=='gui-odyssey'}
local=threading.local()
def archive():
 if not hasattr(local,'z'):local.z=zipfile.ZipFile(Z)
 return local.z
def check(r):
 z=archive();n=r['current_original_step'];a=targets[r['id']]
 out=dict(id=r['id'],source='gui-odyssey',episode_id=r['episode_id'],step=r['step'],approved_for_training=False,fm_approved=False,annotation_path=r['annotation_path'],annotation_sha256=r['annotation_sha256'])
 name='screenshots/'+n['screenshot'];b=z.read(name);info=z.getinfo(name)
 with Image.open(io.BytesIO(b)) as original,Image.open(images[r['id']]['image']) as published:
  x=original.convert('RGB');y=published.convert('RGB')
  out.update(current_pixel_identity_verified=x.size==y.size and x.tobytes()==y.tobytes(),original_size=list(x.size),published_size=list(y.size),
   original_image_sha256=hashlib.sha256(b).hexdigest(),original_rgb_sha256=hashlib.sha256(x.tobytes()).hexdigest(),original_archive_entry=name,original_archive_crc32=info.CRC)
 k=n['action'];v=n['info'];kind=a['action_type']
 if k in ('CLICK','LONG_PRESS') and isinstance(v,list):
  expected='Click' if k=='CLICK' else 'LongPress'
  if kind!=expected:out['action_status']='type_changed'
  elif not isinstance(a['point_2d'],list):out['action_status']='point_missing'
  else:
   out['original_point']=v[0];out['published_point']=a['point_2d']
   out['point_delta']=max(abs(a['point_2d'][i]-v[0][i]) for i in range(2))
   out['action_status']='exact_point' if out['point_delta']==0 else 'point_changed'
 elif k=='TEXT':out['action_status']='exact_text' if kind=='Write' and a['value']==v else 'text_or_type_changed'
 elif k=='CLICK' and isinstance(v,str):
  expected={'KEY_BACK':'NavigateBack','KEY_HOME':'NavigateHome','KEY_ENTER':'KeyboardPress'}.get(v)
  out['action_status']='exact_key' if kind==expected and (v!='KEY_ENTER' or str(a['value']).lower() in ('enter','return')) else 'key_or_type_changed'
 elif k=='COMPLETE':out['action_status']='terminal_type_match' if kind=='Terminate' else 'terminal_rewritten'
 elif k=='SCROLL':out['action_status']='gesture_parameters_require_review'
 else:out['action_status']='unmapped_action'
 nxt=r.get('next_original_step')
 if nxt:
  nextname='screenshots/'+nxt['screenshot'];nb=z.read(nextname)
  out.update(next_archive_entry=nextname,next_original_sha256=hashlib.sha256(nb).hexdigest(),next_step=nxt['step'],next_step_consecutive=nxt['step']==r['step']+1)
 return out
with (P/'odyssey-original-image-action.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 dest=P/'artifacts/odyssey-original-image-action-audit.jsonl';done={}
 if dest.exists():
  for r in rows(dest):done[r['id']]=r
 pending=[r for r in rows(P/'artifacts/odyssey-train-original-evidence-candidates.jsonl') if r['id'] not in done]
 counts=collections.Counter(r['action_status'] for r in done.values());pixel=sum(r['current_pixel_identity_verified'] for r in done.values())
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool,dest.open('a') as out:
  for i,r in enumerate(pool.map(check,pending)):
   out.write(json.dumps(r,ensure_ascii=False)+'\n');counts[r['action_status']]+=1;pixel+=r['current_pixel_identity_verified']
   if i%200==0:
    out.flush();(P/'artifacts/odyssey-original-image-action-progress.json').write_text(json.dumps(dict(done=len(done)+i+1,total=16999,pixel_identity_verified=pixel,actions=counts)))
 report=dict(rows=len(done)+len(pending),pixel_identity_verified=pixel,actions=counts,original_archive=str(Z),original_archive_size=Z.stat().st_size,
  coordinate_basis='Official dataset card: coordinates normalized 0..1000',dataset_card_sha256=hashlib.sha256((Z.parent.parent/'README.md').read_bytes()).hexdigest(),training_ready=False)
 (P/'artifacts/odyssey-original-image-action-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
