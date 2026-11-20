import pathlib,json,io,hashlib,fcntl,collections
import pyarrow.parquet as pq
import numpy as np,cv2
from PIL import Image
cv2.setNumThreads(2)
P=pathlib.Path(__file__).resolve().parent
lock=(P/'mind2web-padding-audit.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
ref={r['id']:r for r in map(json.loads,(P/'artifacts/mind2web-crop-refinement.jsonl').open()) if r['status']=='requires_visual_review'}
old={r['id']:r for r in map(json.loads,(P/'artifacts/mind2web-original-image-registration.jsonl').open()) if r['id'] in ref}
ims={r['id']:r for r in map(json.loads,(P/'artifacts/image-decode-audit.jsonl').open()) if r['id'] in ref}
wanted={(r['episode_id'],r['action_uid']):r for r in old.values()}
dest=P/'artifacts/mind2web-padding-audit.jsonl'
done={r['id']:r for r in map(json.loads,dest.open())} if dest.exists() else {}
rawdir=P/'originals/mind2web-review-images';rawdir.mkdir(exist_ok=True)
def report(final=False):
 obj=dict(completed=len(done),total=len(old),counts=dict(collections.Counter(r['status'] for r in done.values())),training_ready=False,note='Padding geometry diagnostic only; action/instruction/transition gates remain.')
 (P/('artifacts/mind2web-padding-'+('report' if final else 'progress')+'.json')).write_text(json.dumps(obj,indent=2))
with dest.open('a') as out:
 for f in sorted({r['original_shard'] for r in old.values()}):
  for batch in pq.ParquetFile(f).iter_batches(batch_size=4,columns=['annotation_id','action_uid','screenshot']):
   for row in batch.to_pylist():
    prev=wanted.get((row['annotation_id'],row['action_uid']))
    if not prev or prev['id'] in done:continue
    result=dict(id=prev['id'],approved_for_training=False,fm_approved=False)
    try:
     raw=row['screenshot']['bytes'];assert hashlib.sha256(raw).hexdigest()==prev['original_image_sha256']
     rawpath=rawdir/(prev['original_image_sha256']+'.img')
     if not rawpath.exists():rawpath.write_bytes(raw)
     a=np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'));b=np.asarray(Image.open(ims[prev['id']]['image']).convert('RGB'))
     result['original_local_image']=str(rawpath)
     black=(b.max(axis=2)<=8).mean(axis=1)>.995
     n=0
     for v in black[::-1]:
      if not v:break
      n+=1
     result['trailing_nearblack_rows']=n
     if n==0 or len(b)-n<100:result['status']='no_supported_bottom_padding'
     else:
      h=len(b)-n;target=b[:h]
      ga=cv2.resize(cv2.cvtColor(a,cv2.COLOR_RGB2GRAY),(256,len(a)))
      gb=cv2.resize(cv2.cvtColor(target,cv2.COLOR_RGB2GRAY),(256,h))
      scores=cv2.matchTemplate(ga,gb,cv2.TM_SQDIFF_NORMED).ravel()
      candidates=set()
      for v in np.argsort(scores)[:8]:
       candidates.update(range(max(0,int(v)-3),min(len(scores)-1,int(v)+3)+1))
      mae,top=min((float(np.abs(a[t:t+h].astype(np.int16)-target.astype(np.int16)).mean()),t) for t in candidates)
      crop=a[top:top+h].astype(np.float64);tar=target.astype(np.float64)
      score=float(np.square(crop-tar).sum()/max(1.,np.sqrt(np.square(crop).sum()*np.square(tar).sum())))
      result.update(crop_top=top,nonpadding_height=h,content_rgb_mae=mae,content_sqdiff=score,original_remaining_height=len(a)-top,padding_rgb_mae=float(b[h:].mean()),status='low_error_content_with_bottom_padding' if mae<4 and score<.001 else 'padding_content_still_requires_review')
    except Exception as e:result.update(status='error',error=repr(e))
    done[result['id']]=result;out.write(json.dumps(result)+'\n');out.flush();report()
report(True)
