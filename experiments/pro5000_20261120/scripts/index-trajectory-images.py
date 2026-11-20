import hashlib,json,concurrent.futures,time,sqlite3
from pathlib import Path
out=Path('/data/mai/trajectories/20261111');out.mkdir(parents=True,exist_ok=True)
roots=[Path('/data/mai/assets/gui-libra-81k-sft-20261028-v1/extracted'),Path('/data/mai/assets/training-original/audited-originals'),Path('/data/mai/assets/source-layout')]
paths=[p for root in roots for p in root.rglob('*') if p.is_file() and p.suffix.lower() in {'.png','.jpg','.jpeg','.webp','.img'}]
print('IMAGE_FILES',len(paths),flush=True)
def hashfile(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1024*1024):h.update(b)
 return {'sha256':h.hexdigest(),'path':str(p),'size':p.stat().st_size}
with (out/'existing-images.jsonl.partial').open('w') as f,concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
 for i,r in enumerate(pool.map(hashfile,paths)):
  f.write(json.dumps(r)+'\n')
  if (i+1)%10000==0:print('HASHED',i+1,flush=True)
(out/'existing-images.jsonl.partial').replace(out/'existing-images.jsonl')
(out/'existing-images.complete.json').write_text(json.dumps({'files':len(paths),'completed':time.time()}))
print('COMPLETE',flush=True)
