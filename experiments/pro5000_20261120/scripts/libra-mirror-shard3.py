from pathlib import Path
import concurrent.futures,subprocess,hashlib,os
p=Path('/data/mai/benchmarks/gui-libra-base/mirror-model');p.mkdir(exist_ok=True)
name='model-00003-of-00004.safetensors';size=4930760552
url='https://hf-mirror.com/GUI-Libra/GUI-Libra-8B/resolve/15ee285bc1e279eee1b7ea5e953b7dd823e6dbd3/'+name
env=dict(os.environ);env.update(NO_PROXY='*',no_proxy='*')
def part(i):
 a=size*i//8;b=size*(i+1)//8-1;f=p/(name+'.part'+str(i))
 subprocess.run(['curl','-fL','--retry','3','--connect-timeout','30','--max-time','1800','-sS','-r',f'{a}-{b}',url+'?part='+str(i),'-o',str(f)],env=env,check=True)
 assert f.stat().st_size==b-a+1
 print('PART',i,'COMPLETE',flush=True);return f
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:parts=list(pool.map(part,range(8)))
f=p/(name+'.incoming');h=hashlib.sha256()
with f.open('wb') as out:
 for chunk in parts:
  with chunk.open('rb') as src:
   for block in iter(lambda:src.read(8*1024*1024),b''):h.update(block);out.write(block)
assert h.hexdigest()=='580d7849fead7b42edb0d63ae2ef6ab14ae6fa99dca601a9f452274f33f12014'
f.rename(p/name);print('VERIFIED_SHARD_3',flush=True)
