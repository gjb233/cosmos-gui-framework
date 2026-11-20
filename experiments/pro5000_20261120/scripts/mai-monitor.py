import json,re
from pathlib import Path
root=Path('/data/mai/assets/gui-libra-81k-sft-20261028-v1')
s=json.loads((root/'download-status.json').read_text());rows=[]
for p in sorted(root.glob('*.transfer.log')):
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-20000));t=f.read().decode(errors='replace')
 m=re.findall(r'\[#[^\]]+\]',t)
 rows.append({'file':p.name,'progress':m[-1] if m else ''})
ready=root.parent/'training-ready/data-readiness.json'
print(json.dumps({'stage':s['stage'],'verified_files':len(s['verified']),'errors':s['errors'],'progress':rows,'extraction_complete':(root/'extraction-status.json').exists(),'readiness':json.loads(ready.read_text()) if ready.exists() else None}))
