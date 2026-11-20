from pathlib import Path
import time,hashlib,concurrent.futures
b=Path('/data/mai/benchmarks/gui-libra-base');m=b/'mirror-model'
def assemble(i,size,sha):
 name=f'model-{i:05d}-of-00004.safetensors';dest=m/name
 while not dest.exists():
  src=b/'cached-model'/(name+'.incoming')
  if not src.exists():src=b/'cached-model'/name
  prefix=src.stat().st_size if src.exists() else 0
  segments=[]
  for j in range(8):
   a=size*j//8;z=size*(j+1)//8;part=m/(name+'.part'+str(j))
   if prefix>=z:segments.append((src,a,z-a))
   elif part.exists() and part.stat().st_size==z-a:segments.append((part,0,z-a))
   else:break
  if len(segments)!=8:time.sleep(10);continue
  tmp=m/(name+'.stitched');h=hashlib.sha256()
  with tmp.open('wb') as out:
   for p,offset,n in segments:
    with p.open('rb') as f:
     f.seek(offset)
     while n:
      block=f.read(min(n,8*1024*1024));assert block;n-=len(block);h.update(block);out.write(block)
  assert h.hexdigest()==sha,'stitched SHA mismatch'
  if not dest.exists():tmp.rename(dest)
  print('VERIFIED_STITCHED',name,flush=True)
  return
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 futures=[pool.submit(assemble,2,4994158816,'532a7459642847e41d1816049251c0846710610bb3b01dcec4d180bd3e4c9043'),pool.submit(assemble,3,4930760552,'580d7849fead7b42edb0d63ae2ef6ab14ae6fa99dca601a9f452274f33f12014')]
 for f in futures:f.result()
