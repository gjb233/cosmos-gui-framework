import pathlib,json,urllib.request,hashlib,concurrent.futures,fcntl,time
P=pathlib.Path(__file__).resolve().parent;O=P/'originals/multimodal-mind2web-train';O.mkdir(exist_ok=True)
lock=(P/'multimodal-mind2web-download.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
info=json.loads((P/'originals/multimodal-mind2web-tree.json').read_text())
items=[x for x in info['files'] if x['path'].startswith('data/train-')]
def worker(x):
 path=O/pathlib.Path(x['path']).name
 proof=path.with_suffix('.verified.json')
 if path.exists() and proof.exists():return json.loads(proof.read_text())
 u='https://huggingface.co/datasets/'+info['repo']+'/resolve/'+info['revision']+'/'+x['path']
 for attempt in range(3):
  try:
   sha=hashlib.sha256();n=0;tmp=path.with_suffix('.partial')
   with urllib.request.urlopen(u,timeout=120) as r,tmp.open('wb') as f:
    while True:
     b=r.read(4*1024*1024)
     if not b:break
     f.write(b);sha.update(b);n+=len(b)
   assert n==x['size'] and sha.hexdigest()==x['lfs']['oid']
   tmp.replace(path);out=dict(path=str(path),size=n,sha256=sha.hexdigest(),revision=info['revision'],status='verified')
   proof.write_text(json.dumps(out));return out
  except Exception as e:
   if attempt==2:return dict(path=str(path),status='error',error=repr(e))
   time.sleep(5*(attempt+1))
done=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 for future in concurrent.futures.as_completed([pool.submit(worker,x) for x in items]):
  done.append(future.result());(O/'progress.json').write_text(json.dumps(dict(completed=len(done),total=len(items),results=done),indent=2));print(len(done),flush=True)
(O/'complete.json').write_text(json.dumps(dict(results=done,total=len(items),failures=sum(x['status']=='error' for x in done)),indent=2))
