import pathlib,json,zipfile,hashlib,io,threading,concurrent.futures,collections,fcntl,time
from PIL import Image
P=pathlib.Path(__file__).resolve().parent
lock=(P/'amex-original-image-audit.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
Z=pathlib.Path('/rscratch/wzzheng/data/mai/OS-Atlas-data/mobile_domain/amex_images.zip')
images={r['id']:r for r in map(json.loads,(P/'artifacts/image-decode-audit.jsonl').open()) if r['source']=='amex'}
rows=list(map(json.loads,(P/'artifacts/amex-original-action-candidates.jsonl').open()))
out=P/'artifacts/amex-original-image-audit.jsonl';done={}
if out.exists():
 for r in map(json.loads,out.open()):done[r['id']]=r
local=threading.local()
def check(r):
 if not hasattr(local,'z'):local.z=zipfile.ZipFile(Z)
 z=local.z;n=r['matches'][0]['original_step'];name='screenshots/'+n['image_path']
 result=dict(id=r['id'],source='amex',episode_id=r['episode_id'],step=r['step'],original_step_id=n['step_id'],archive=str(Z),entry=name,annotation_sha256=r['matches'][0]['annotation_sha256'],approved_for_training=False,fm_approved=False)
 try:
  b=z.read(name)
  with Image.open(io.BytesIO(b)) as orig,Image.open(images[r['id']]['image']) as pub:
   x=orig.convert('RGB');y=pub.convert('RGB')
   result.update(original_size=list(x.size),published_size=list(y.size),current_pixel_identity_verified=x.size==y.size and x.tobytes()==y.tobytes(),original_image_sha256=hashlib.sha256(b).hexdigest(),original_rgb_sha256=hashlib.sha256(x.tobytes()).hexdigest(),archive_crc32=z.getinfo(name).CRC)
  nxt=r['matches'][0].get('next_original_step')
  if nxt:
   nn='screenshots/'+nxt['image_path'];nb=z.read(nn)
   with Image.open(io.BytesIO(nb)) as im:im.load();ns=list(im.size)
   result.update(next_entry=nn,next_image_sha256=hashlib.sha256(nb).hexdigest(),next_size=ns,next_step_consecutive=nxt['step_id']==n['step_id']+1)
  result['status']='checked'
 except Exception as e:result.update(status='error',error=repr(e))
 return result
def report(final=False):
 c=collections.Counter(r['status'] for r in done.values())
 obj=dict(completed=len(done),total=len(rows),counts=c,pixel_identity_verified=sum(r.get('current_pixel_identity_verified',False) for r in done.values()),next_consecutive=sum(r.get('next_step_consecutive',False) for r in done.values()),training_ready=False,archive=str(Z),note='Original named screenshots from OS-Atlas AMEX mirror; pixel/CRC/SHA evidence only, source archive provenance and split/action approval still required.')
 (P/('artifacts/amex-original-image-report.json' if final else 'artifacts/amex-original-image-progress.json')).write_text(json.dumps(obj,indent=2))
with out.open('a') as f,concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for r in pool.map(check,[r for r in rows if r['id'] not in done]):
  done[r['id']]=r;f.write(json.dumps(r)+'\n')
  if len(done)%100==0:f.flush();report()
report(True)
print('COMPLETE',len(done),flush=True)
