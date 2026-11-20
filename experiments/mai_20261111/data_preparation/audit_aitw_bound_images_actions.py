import pathlib,json,hashlib,fcntl,collections,concurrent.futures,math
from PIL import Image,ImageChops,ImageStat
P=pathlib.Path(__file__).resolve().parent
lock=(P/'aitw-bound-image-action.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
images={r['id']:r for r in map(json.loads,(P/'artifacts/image-decode-audit.jsonl').open()) if r['source']=='aitw'}
actions={r['id']:r.get('action',{}) for r in map(json.loads,(P/'artifacts/action-pair-audit.jsonl').open()) if r['source']=='aitw'}
rows=[r for r in map(json.loads,(P/'artifacts/aitw-official-incremental-bindings.jsonl').open()) if r['status']=='unique_original_record']
dest=P/'artifacts/aitw-bound-image-action-audit.jsonl'
done={r['id']:r for r in map(json.loads,dest.open())} if dest.exists() else {}
def signature(r):
 return hashlib.sha256(json.dumps([r['original_record']['raw_record_sha256'],[n['raw_record_sha256'] for n in r['next_original_records']]],sort_keys=True).encode()).hexdigest()
def original_image(o):
 f=o['features'];w=f['image/width'][0];h=f['image/height'][0];c=f['image/channels'][0]
 raw=pathlib.Path(o['image_path']).read_bytes()
 assert c==3 and len(raw)==w*h*c,'Unexpected raw RGB shape'
 return Image.frombytes('RGB',(w,h),raw)
def work(r):
 o=r['original_record'];f=o['features'];a=actions.get(r['id'],{})
 result=dict(id=r['id'],binding_signature=signature(r),episode_id=r['original_episode_id'],step=r['step'],split=r['official_parent_split'],approved_for_training=False,fm_approved=False)
 try:
  assert hashlib.sha256(pathlib.Path(o['raw_record_path']).read_bytes()).hexdigest()==o['raw_record_sha256']
  raw=pathlib.Path(o['image_path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==o['image_sha256']
  with original_image(o) as x,Image.open(images[r['id']]['image']) as y:
   x=x.convert('RGB');y=y.convert('RGB')
   result.update(original_size=list(x.size),published_size=list(y.size),pixel_identity_verified=x.size==y.size and x.tobytes()==y.tobytes())
   if x.size==y.size:result['rgb_mean_absolute_error']=sum(ImageStat.Stat(ImageChops.difference(x,y)).mean)/3
  ns=r['next_original_records'];result['unique_consecutive_next']=len(ns)==1 and ns[0]['episode_id']==o['episode_id'] and ns[0]['step']==o['step']+1
  if result['unique_consecutive_next']:
   n=ns[0];assert hashlib.sha256(pathlib.Path(n['image_path']).read_bytes()).hexdigest()==n['image_sha256']
   with original_image(n) as im:im.load()
   result['next_image_path']=n['image_path'];result['next_image_sha256']=n['image_sha256']
  typ=f['results/action_type'][0];result['raw_action_type']=typ;result['published_action_type']=a.get('action_type')
  if typ==3:
   result['action_diagnostic']='exact_type_text' if a.get('action_type')=='Write' and a.get('value')==f.get('results/type_action',[''])[0] else 'type_or_text_differs'
  elif typ==4:
   touch=f['results/yx_touch'];lift=f['results/yx_lift'];d=math.dist(touch,lift)
   result.update(touch_yx=touch,lift_yx=lift,touch_lift_distance=d,action_diagnostic='dual_point_requires_target_or_gesture_review')
  else:
   result['action_diagnostic']='key_or_terminal_requires_parameter_review'
  result['status']='checked'
 except Exception as e:result.update(status='error',error=repr(e))
 return result
todo=[r for r in rows if r['id'] not in done or done[r['id']].get('status')=='error' or done[r['id']].get('binding_signature')!=signature(r)]
def report():
 obj=dict(bound_records=len(rows),completed=len(done),pixel_identity_verified=sum(r.get('pixel_identity_verified',False) for r in done.values()),counts=dict(collections.Counter(r['status'] for r in done.values())),action_diagnostics=dict(collections.Counter(r.get('action_diagnostic') for r in done.values())),training_ready=False)
 (P/'artifacts/aitw-bound-image-action-report.json').write_text(json.dumps(obj,indent=2))
with dest.open('a') as out,concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for r in pool.map(work,todo):
  done[r['id']]=r;out.write(json.dumps(r)+'\n')
  if len(done)%100==0:out.flush();report()
report()
