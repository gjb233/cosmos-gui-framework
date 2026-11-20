import pathlib,json,hashlib,fcntl,concurrent.futures,urllib.request,datetime
P=pathlib.Path('/rscratch/wzzheng/experiments/mixed-sft-preparation-20261028-v1')
O=pathlib.Path('/rscratch/wzzheng/experiments/guiodyssey-high-a27-20261028-v1')
out=P/'originals/odyssey-train';out.mkdir(exist_ok=True)
lock=(out/'download.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
rev=json.load(open(O/'dataset-revision.json'))
tree={x['path']:x for x in json.load(open(O/'dataset-files.json')) if x['type']=='file'}
train=set(json.load(open(O/'data/splits/random_split.json'))['train'])
need={r['image'].rsplit('_',1)[0]+'.json' for r in map(json.loads,open(P/'artifacts/inventory.jsonl')) if r['source']=='gui-odyssey'}&train
def work(name):
 path='annotations/'+name; meta=tree.get(path);dest=out/name
 result=dict(episode_file=name,revision=rev['sha'],repo=rev['repo'])
 try:
  assert meta is not None,'missing_in_pinned_tree'
  raw=dest.read_bytes() if dest.exists() else urllib.request.urlopen('https://huggingface.co/datasets/'+rev['repo']+'/resolve/'+rev['sha']+'/'+path,timeout=45).read()
  assert len(raw)==meta['size']
  oid=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
  assert oid==meta['oid'],'git_blob_mismatch'
  data=json.loads(raw);assert str(data['episode_id'])==name[:-5]
  if not dest.exists():
   tmp=dest.with_suffix('.tmp');tmp.write_bytes(raw);tmp.replace(dest)
  result.update(status='verified',sha256=hashlib.sha256(raw).hexdigest(),path=str(dest),steps=len(data['steps']))
 except Exception as e:result.update(status='failed',error=repr(e))
 return result
counts={'verified':0,'failed':0}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool,(out/'download-results.jsonl').open('w') as f:
 for r in pool.map(work,sorted(need)):
  f.write(json.dumps(r)+'\n');f.flush();counts[r['status']]+=1
  (out/'progress.json').write_text(json.dumps(dict(total=len(need),**counts)))
(out/'complete.json').write_text(json.dumps(dict(total=len(need),**counts,training_ready=False,time=datetime.datetime.now(datetime.timezone.utc).isoformat())))
