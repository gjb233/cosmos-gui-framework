import pathlib,json,urllib.request,hashlib,time,concurrent.futures,fcntl
P=pathlib.Path(__file__).resolve().parent;O=P/'originals'
lock=(P/'original-source-download.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
tree=json.loads((O/'guiact-tree.json').read_text())
m=next(x for x in tree if x['path']=='web-multi_train_images.parquet')
jobs=[dict(name='guiact-web-multi-train-images.parquet',url='https://huggingface.co/datasets/yiye2023/GUIAct/resolve/9f3341d4670624aa6bf4a6eb71544d1b09961c0e/'+m['path'],size=m['size'],expected_sha256=m['lfs']['oid'],source='yiye2023/GUIAct',revision='9f3341d4670624aa6bf4a6eb71544d1b09961c0e'),
dict(name='coat-author-original.zip',url='https://drive.usercontent.google.com/download?id=12xOV2m62fBUFLhMcWIsFiC6zwV7a2RhI&export=download&confirm=t',size=2645279982,source='IMNearth/CoAT author README Google Drive link')]
def run(j):
 path=O/j['name'];proof=path.with_suffix(path.suffix+'.provenance.json')
 if path.exists() and proof.exists():return json.loads(proof.read_text())
 tmp=path.with_suffix(path.suffix+'.partial');h=hashlib.sha256();size=0
 with urllib.request.urlopen(j['url'],timeout=120) as response,tmp.open('wb') as f:
  content_type=response.headers.get('Content-Type')
  while True:
   b=response.read(4*1024*1024)
   if not b:break
   f.write(b);h.update(b);size+=len(b)
 assert size==j['size'],(size,j['size'])
 sha=h.hexdigest()
 if j.get('expected_sha256'):assert sha==j['expected_sha256']
 tmp.replace(path);result=dict(j,path=str(path),sha256=sha,downloaded_bytes=size,content_type=content_type,verified_size=True,upstream_hash_verified=bool(j.get('expected_sha256')))
 proof.write_text(json.dumps(result,indent=2));return result
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 for fu in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):
  try:r=fu.result()
  except Exception as e:r=dict(status='error',error=repr(e))
  results.append(r);(O/'additional-source-download-progress.json').write_text(json.dumps(results,indent=2));print(json.dumps(r),flush=True)
(O/'additional-source-download-complete.json').write_text(json.dumps(results,indent=2))
