import tomllib
from pathlib import Path
lock=tomllib.loads(Path('/data/mai/archives/uv.lock.original').read_text())
names=['webdataset','iopath','peft','qwen-vl-utils','wandb','timm','einx','ftfy','opencv-python','sentencepiece','tensorstore']
lines=[]
for name in names:
    matches=[p for p in lock['package'] if p['name']==name]
    if len(matches)==1: lines.append(name+'=='+matches[0]['version'])
Path('/data/mai/mai-extra-requirements.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
