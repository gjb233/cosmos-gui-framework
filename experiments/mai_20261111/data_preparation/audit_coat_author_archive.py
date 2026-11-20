import pathlib,json,zipfile,hashlib,collections,fcntl,io
from PIL import Image
P=pathlib.Path(__file__).resolve().parent
lock=(P/'coat-author-archive-audit.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
Z=P/'originals/coat-author-original.zip'
images={r['id']:r for r in map(json.loads,(P/'artifacts/image-decode-audit.jsonl').open()) if r['source']=='coat-terminal'}
counts=collections.Counter();records=[]
with zipfile.ZipFile(Z) as z:
 names=z.namelist();idx=collections.defaultdict(list)
 for name in names:
  if '/train/' in name:idx['train/'+name.split('/train/',1)[1]].append(name)
 with (P/'artifacts/coat-author-archive-audit.jsonl').open('w') as out:
  for r in map(json.loads,(P/'artifacts/original-trajectory-candidates.jsonl').open()):
   if r['source']!='coat-terminal':continue
   suffix='train/'+r['current_image'].split('/train/',1)[1]
   annotation='train/'+r['annotation_path'].split('/train/',1)[1]
   row=dict(id=r['id'],source=r['source'],episode_id=r['episode_id'],step=r['step'],approved_for_training=False,fm_approved=False)
   try:
    assert len(idx[suffix])==1 and len(idx[annotation])==1,'nonunique_author_archive_entry'
    image_entry=idx[suffix][0];ann_entry=idx[annotation][0]
    b=z.read(image_entry);ann=z.read(ann_entry);sha=hashlib.sha256(b).hexdigest()
    same=sha==images[r['id']]['sha256']
    if not same:
     with Image.open(io.BytesIO(b)) as a,Image.open(r['current_image']) as pub:
      same=a.size==pub.size and a.convert('RGB').tobytes()==pub.convert('RGB').tobytes()
    row.update(status='checked',author_train_archive_entry=image_entry,author_annotation_entry=ann_entry,current_image_identity_verified=same,original_image_sha256=sha,original_annotation_sha256=hashlib.sha256(ann).hexdigest(),annotation_bytes_identical=hashlib.sha256(ann).hexdigest()==r['annotation_sha256'])
    nxt=r.get('next_image_candidate')
    if nxt:
     ns='train/'+nxt.split('/train/',1)[1]
     if len(idx[ns])==1:
      nb=z.read(idx[ns][0]);row.update(next_author_entry=idx[ns][0],next_image_sha256=hashlib.sha256(nb).hexdigest())
   except Exception as e:row.update(status='error',error=repr(e))
   counts[row['status']]+=1
   counts['current_identity_verified']+=int(row.get('current_image_identity_verified',False))
   counts['annotation_identical']+=int(row.get('annotation_bytes_identical',False))
   counts['next_present_in_author_archive']+=int('next_author_entry' in row)
   out.write(json.dumps(row)+'\n')
   if sum(counts[k] for k in ['checked','error'])%100==0:
    out.flush();(P/'artifacts/coat-author-archive-progress.json').write_text(json.dumps(dict(counts=counts)))
report=dict(counts=counts,training_ready=False,source_archive=str(Z),source_provenance=str(Z)+'.provenance.json',note='Author archive train membership and byte/pixel identity only; action and instruction/next alignment approval remains separate.')
(P/'artifacts/coat-author-archive-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
