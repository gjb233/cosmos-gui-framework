import pathlib,json,collections,io,hashlib,fcntl
import pyarrow.parquet as pq
import numpy as np,cv2
from PIL import Image,ImageChops,ImageStat
cv2.setNumThreads(2)
P=pathlib.Path(__file__).resolve().parent
lock=(P/'mind2web-original-image-audit.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
O=P/'originals/multimodal-mind2web-train'
images={pathlib.Path(r['image']).stem:r for r in map(json.loads,(P/'artifacts/image-decode-audit.jsonl').open()) if r['source']=='mind2web'}
dest=P/'artifacts/mind2web-original-image-registration.jsonl'
done={}
if dest.exists():
 for r in map(json.loads,dest.open()):done[r['id']]=r
def progress(final=False):
 c=collections.Counter(r['status'] for r in done.values())
 (P/('artifacts/mind2web-original-image-report.json' if final else 'artifacts/mind2web-original-image-progress.json')).write_text(json.dumps(dict(completed=len(done),total=len(images),counts=c,training_ready=False,note='Image registration/crop evidence only. Lossy JPEG matches are not byte identity; original viewport/transition approval is separate.'),indent=2))
with dest.open('a') as out:
 for f in sorted(O.glob('*.parquet')):
  proof=json.loads(f.with_suffix('.verified.json').read_text())
  for batch in pq.ParquetFile(f).iter_batches(batch_size=4,columns=['annotation_id','action_uid','screenshot','confirmed_task','target_action_index','operation']):
   for r in batch.to_pylist():
    key=r['annotation_id']+'-'+r['action_uid']
    if key not in images:continue
    im=images[key]
    if im['id'] in done:continue
    result=dict(id=im['id'],source='mind2web',episode_id=r['annotation_id'],action_uid=r['action_uid'],original_shard=str(f),original_shard_sha256=proof['sha256'],original_step=r['target_action_index'],original_goal=r['confirmed_task'],original_operation=r['operation'],approved_for_training=False,fm_approved=False)
    try:
     rawbytes=r['screenshot']['bytes']
     with Image.open(io.BytesIO(rawbytes)) as original,Image.open(im['image']) as published:
      x=original.convert('RGB');y=published.convert('RGB')
      result.update(original_size=list(x.size),published_size=list(y.size),original_image_sha256=hashlib.sha256(rawbytes).hexdigest())
      if x.size==y.size and x.tobytes()==y.tobytes():result.update(status='exact_rgb_identity',crop_top=0)
      elif x.width==y.width and x.height>=y.height:
       xx=np.asarray(x);yy=np.asarray(y)
       a=cv2.resize(cv2.cvtColor(xx,cv2.COLOR_RGB2GRAY),None,fx=.25,fy=.25)
       b=cv2.resize(cv2.cvtColor(yy,cv2.COLOR_RGB2GRAY),None,fx=.25,fy=.25)
       res=cv2.matchTemplate(a,b,cv2.TM_SQDIFF_NORMED);score,_,loc,_=cv2.minMaxLoc(res)
       guess=int(loc[1]*4);best=None
       for top in range(max(0,guess-8),min(x.height-y.height,guess+8)+1):
        crop=x.crop((0,top,y.width,top+y.height));mae=sum(ImageStat.Stat(ImageChops.difference(crop,y)).mean)/3
        if best is None or mae<best[0]:best=(mae,top)
       mae,top=best;crop=x.crop((0,top,y.width,top+y.height))
       result.update(status='lossy_crop_registered' if mae<4 and score<.001 else 'crop_registration_requires_review',crop_top=top,rgb_mean_absolute_error=mae,template_normalized_sqdiff=score,exact_crop_rgb=crop.tobytes()==y.tobytes())
       if result['exact_crop_rgb']:result['status']='exact_rgb_crop'
      else:result['status']='size_transform_requires_review'
    except Exception as e:result.update(status='error',error=repr(e))
    out.write(json.dumps(result,ensure_ascii=False)+'\n');done[result['id']]=result
    if len(done)%50==0:out.flush();progress()
progress(True);print('COMPLETE',len(done),flush=True)
