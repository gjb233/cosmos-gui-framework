import tomllib
from pathlib import Path
lock=tomllib.loads(Path('/data/mai/archives/uv.lock.original').read_text())
names=['pandas','h5py','aiofiles','blobfile','fvcore','datasets','huggingface-hub','transformers']
lines=[]
for name in names:
    versions=sorted(set(p['version'] for p in lock['package'] if p['name']==name))
    print(name,versions)
    if len(versions)==1: lines.append(name+'=='+versions[0])
Path('/data/mai/data-pins.txt').write_text('\n'.join(lines)+'\n')
