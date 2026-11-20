import pathlib,json,collections,io,hashlib,fcntl
import pyarrow.parquet as pq
import numpy as np,cv2
from PIL import Image
cv2.setNumThreads(2)
P=pathlib.Path(__file__).resolve().parent
lock=(P/'mind2web-crop-refine.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
old={r['id']:r for r in map(json.loads,(P/'artifacts/mind2web-original-image-registration.jsonl').open()) if r['status']=='crop_registration_requires_review'}
ims={r['id']:r for r in map(json.loads,(P/'artifacts/image-decode-audit.jsonl').open()) if r['id'] in old}
wanted={(r['episode_id'],r['action_uid']):r for r in old.values()}
dest=P/'artifacts/mind2web-crop-refinement.jsonl'
done={r['id']:r for r in map(json.loads,dest.open())} if dest.exists() else {}
def report(final=False):
 obj=dict(completed=len(done),total=len(old),counts=dict(collections.Counter(x['status'] for x in done.values())),training_ready=False,note='Refined screenshot geometry evidence; no automatic action, instruction or FM approval.')
 (P/('artifacts/mind2web-crop-refinement-report.json' if final else 'artifacts/mind2web-crop-refinement-progress.json')).write_text(json.dumps(obj,indent=2))
with dest.open('a') as out:
 for f in sorted({r['original_shard'] for r in old.values()}):
  for batch in pq.ParquetFile(f).iter_batches(batch_size=4,columns=['annotation_id','action_uid','screenshot']):
   for row in batch.to_pylist():
    prev=wanted.get((row['annotation_id'],row['action_uid']))
    if not prev or prev['id'] in done:continue
    result=dict(id=prev['id'],previous_crop_top=prev['crop_top'],previous_mae=prev['rgb_mean_absolute_error'],approved_for_training=False,fm_approved=False)
    try:
     raw=row['screenshot']['bytes'];assert hashlib.sha256(raw).hexdigest()==prev['original_image_sha256']
     a=np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'));b=np.asarray(Image.open(ims[prev['id']]['image']).convert('RGB'))
     assert a.shape[1]==b.shape[1] and a.shape[0]>=b.shape[0]
     # Preserve exact vertical coordinates; only reduce horizontal width for candidate search.
     ga=cv2.resize(cv2.cvtColor(a,cv2.COLOR_RGB2GRAY),(256,a.shape[0]))
     gb=cv2.resize(cv2.cvtColor(b,cv2.COLOR_RGB2GRAY),(256,b.shape[0]))
     scores=cv2.matchTemplate(ga,gb,cv2.TM_SQDIFF_NORMED).ravel()
     candidates=set([prev['crop_top']])
     for v in np.argsort(scores)[:8]:
      candidates.update(range(max(0,int(v)-2),min(len(scores)-1,int(v)+2)+1))
     best=min((float(np.abs(a[t:t+len(b)].astype(np.int16)-b.astype(np.int16)).mean()),t) for t in candidates)
     mae,top=best
     crop=a[top:top+len(b)].astype(np.float64);target=b.astype(np.float64)
     score=float(np.square(crop-target).sum()/max(1.,np.sqrt(np.square(crop).sum()*np.square(target).sum())))
     result.update(crop_top=top,rgb_mean_absolute_error=mae,final_aligned_normalized_sqdiff=score,status='refined_low_error_crop' if mae<4 and score<.001 else 'requires_visual_review')
    except Exception as e:result.update(status='error',error=repr(e))
    done[result['id']]=result;out.write(json.dumps(result)+'\n');out.flush()
    if len(done)%20==0:report()
report(True)
