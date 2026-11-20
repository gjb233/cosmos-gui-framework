import pathlib,json,collections,base64,io,hashlib,fcntl
import pyarrow.parquet as pq
from PIL import Image
P=pathlib.Path(__file__).resolve().parent;O=P/'originals/guiact-selected-images';O.mkdir(exist_ok=True)
lock=(P/'guiact-original-image-audit.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
images=collections.defaultdict(list)
for r in map(json.loads,(P/'artifacts/image-decode-audit.jsonl').open()):
 if r['source'].startswith('guiact'):images[pathlib.Path(r['image']).stem].append(r)
needed=set(images)
for r in map(json.loads,(P/'artifacts/original-trajectory-candidates.jsonl').open()):
 if r['source'].startswith('guiact') and r.get('next_image_candidate'):needed.add(pathlib.Path(r['next_image_candidate']).stem)
source=P/'originals/guiact-web-multi-train-images.parquet';counts=collections.Counter();matched=set();seen=set();scanned=0
with (P/'artifacts/guiact-original-image-audit.jsonl').open('w') as out,(P/'artifacts/guiact-original-image-index.jsonl').open('w') as index:
 for batch in pq.ParquetFile(source).iter_batches(batch_size=64,columns=['base64','__index_level_0__']):
  for raw in batch.to_pylist():
   scanned+=1;uid=raw['__index_level_0__']
   if uid not in needed:continue
   text=raw['base64'];text=text.split(',',1)[1] if text.startswith('data:') else text
   b=base64.b64decode(text);sha=hashlib.sha256(b).hexdigest()
   dest=O/(uid+'-'+sha+'.img');dest.write_bytes(b)
   with Image.open(io.BytesIO(b)) as original:
    original=original.convert('RGB');rgb=original.tobytes()
    index.write(json.dumps(dict(image_uid=uid,image_path=str(dest),sha256=sha,size=list(original.size),original_parquet=str(source)))+'\n')
    for im in images.get(uid,[]):
     row=dict(id=im['id'],source=im['source'],image_uid=uid,original_sha256=sha,original_image_path=str(dest),original_parquet=str(source),approved_for_training=False,fm_approved=False)
     try:
      with Image.open(im['image']) as pub:
       pub=pub.convert('RGB')
       same=original.size==pub.size and rgb==pub.tobytes()
       row.update(status='checked',current_pixel_identity_verified=same,original_size=list(original.size),published_size=list(pub.size))
       if same:matched.add(im['id'])
     except Exception as e:row.update(status='error',error=repr(e))
     counts[row['status']]+=1;out.write(json.dumps(row)+'\n');seen.add(im['id'])
  out.flush();index.flush()
  (P/'artifacts/guiact-original-image-progress.json').write_text(json.dumps(dict(scanned_original_rows=scanned,unique_current_ids_seen=len(seen),unique_pixel_identity_verified=len(matched),total_current=sum(map(len,images.values())),training_ready=False)))
report=dict(scanned_original_rows=scanned,unique_current_ids_seen=len(seen),unique_pixel_identity_verified=len(matched),total_current=sum(map(len,images.values())),missing_current_ids=[im['id'] for ims in images.values() for im in ims if im['id'] not in seen],training_ready=False,note='Pinned official training parquet image identity. Official test rows intentionally not in train parquet; action/next approval remains separate.')
(P/'artifacts/guiact-original-image-report.json').write_text(json.dumps(report,indent=2));print('COMPLETE',len(matched),flush=True)
