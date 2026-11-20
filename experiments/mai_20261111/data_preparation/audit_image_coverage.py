import json, collections, datetime
from pathlib import Path
R=Path(__file__).resolve().parent
D=Path('/rscratch/wzzheng/datasets/gui-libra-81k-sft-20261028-v1/extracted')
index=collections.defaultdict(list)
for p in D.rglob('*'):
    if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}: index[p.name].append(p)
counts=collections.defaultdict(collections.Counter)
resolved=[]
for line in (R/'artifacts/inventory.jsonl').open():
    row=json.loads(line); name=row['image']; candidates=index.get(Path(name).name,[])
    exact=[p for p in candidates if p.as_posix().endswith('/'+name)]
    matches=exact or candidates
    status='unique' if len(matches)==1 else 'missing' if not matches else 'ambiguous'
    counts[row['source']][status]+=1
    if status=='unique': resolved.append({'id':row['id'],'source':row['source'],'image':str(matches[0])})
with (R/'artifacts/image-resolution.jsonl').open('w') as f:
    for r in resolved:f.write(json.dumps(r)+'\n')
report={'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'counts':{s:dict(v) for s,v in counts.items()},'resolved':len(resolved),'training_ready':False,'note':'Path resolution only, not trajectory adjacency, heldout exclusion or image decode validation.'}
(R/'artifacts/image-coverage.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
