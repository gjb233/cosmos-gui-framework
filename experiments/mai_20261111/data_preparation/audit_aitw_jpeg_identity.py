import pathlib,json,io,hashlib,fcntl,collections,concurrent.futures
from PIL import Image
P=pathlib.Path(__file__).resolve().parent
lock=(P/'aitw-jpeg-identity.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
ims={r['id']:r for r in map(json.loads,(P/'artifacts/image-decode-audit.jsonl').open()) if r['source']=='aitw'}
rows=[r for r in map(json.loads,(P/'artifacts/aitw-official-incremental-bindings.jsonl').open()) if r['status']=='unique_original_record']
dest=P/'artifacts/aitw-jpeg-identity.jsonl'
done={r['id']:r for r in map(json.loads,dest.open())} if dest.exists() else {}
def work(r):
 o=r['original_record'];f=o['features'];result=dict(id=r['id'],raw_sha=o['image_sha256'],approved_for_training=False,fm_approved=False)
 try:
  raw=pathlib.Path(o['image_path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==o['image_sha256']
  w=f['image/width'][0];h=f['image/height'][0];assert f['image/channels']==[3] and len(raw)==w*h*3
  im=Image.frombytes('RGB',(w,h),raw);pub=pathlib.Path(ims[r['id']]['image']).read_bytes()
  result['published_sha256']=hashlib.sha256(pub).hexdigest();result['status']='encoding_not_reproduced'
  with Image.open(io.BytesIO(pub)) as target:
   target=target.convert('RGB')
   for q in [75,90,95]:
    b=io.BytesIO();im.save(b,format='JPEG',quality=q)
    encoded=b.getvalue()
    if encoded==pub:
     result.update(status='exact_jpeg_bytes_reproduced',pillow_quality=q);break
    with Image.open(io.BytesIO(encoded)) as rec:
     if rec.size==target.size and rec.convert('RGB').tobytes()==target.tobytes():
      result.update(status='exact_decoded_jpeg_reproduced',pillow_quality=q);break
 except Exception as e:result.update(status='error',error=repr(e))
 return result
def report():
 (P/'artifacts/aitw-jpeg-identity-report.json').write_text(json.dumps(dict(completed=len(done),bound_total=len(rows),counts=dict(collections.Counter(r['status'] for r in done.values())),training_ready=False,note='Exact reproducible encoding evidence only; original action/split/instruction/next approval is separate.'),indent=2))
with dest.open('a') as out,concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for i,r in enumerate(pool.map(work,[r for r in rows if r['id'] not in done]),1):
  done[r['id']]=r;out.write(json.dumps(r)+'\n')
  if i%100==0:out.flush();report()
report()
