import json,time
from pathlib import Path
root=Path('/data/mai/assets/gui-libra-81k-sft-20261028-v1')
last=None
for n in range(480):
 s=json.loads((root/'download-status.json').read_text());ready=root.parent/'training-ready/data-readiness.json'
 used=sum(p.stat().st_blocks*512 for p in (root/'snapshot/data/images').glob('*') if p.is_file() and not p.name.endswith('.aria2'))
 state={'time':time.strftime('%H:%M:%S'),'image_gib':round(used/1024**3,2),'verified':len(s['verified']),'download':s['stage'],'extracted':len(list(root.glob('*.extracted.json'))),'errors':s['errors']}
 if ready.exists():state['readiness']=json.loads(ready.read_text())
 print(json.dumps(state),flush=True)
 if ready.exists() or s['stage']=='failed':break
 time.sleep(45)
