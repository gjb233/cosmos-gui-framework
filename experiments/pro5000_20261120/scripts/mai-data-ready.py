import json,time,collections,shutil,hashlib
from pathlib import Path
base=Path('/data/mai/assets');root=base/'gui-libra-81k-sft-20261028-v1'
for _ in range(2880):
 if (base/'supplemental-assets.complete').exists() and (root/'extraction-status.json').exists():break
 state=json.loads((root/'download-status.json').read_text())
 if state['stage']=='failed':raise SystemExit('Download failed; readiness audit stopped')
 time.sleep(30)
else:raise SystemExit('Assets not complete within 24h')
original=base/'training-original/artifacts';out=base/'training-ready';out.mkdir(exist_ok=True)
maps=[('/rscratch/wzzheng/datasets/gui-libra-81k-sft-20261028-v1/extracted/audited-originals',str(base/'training-original/audited-originals')),('/rscratch/wzzheng/datasets/gui-libra-81k-sft-20261028-v1',str(root)),('/rscratch/wzzheng/experiments',str(base/'source-layout/experiments'))]
refs=set();count=0

def change(x):
 if isinstance(x,dict):return {k:change(v) for k,v in x.items()}
 if isinstance(x,list):return [change(v) for v in x]
 if isinstance(x,str):
  for old,new in maps:
   if x==old or x.startswith(old+'/'):
    x=new+x[len(old):];refs.add(x);return x
 return x
with (original/'train-manifest.jsonl').open() as src,(out/'train-manifest.jsonl').open('w') as dst:
 for line in src:
  row=json.loads(line);updated=change(row);assert row['id']==updated['id'];count+=1
  dst.write(json.dumps(updated,ensure_ascii=False)+'\n')
schedules={}
for name in ['schedule-phase1.jsonl','schedule-phase2.jsonl']:
 shutil.copyfile(original/name,out/name)
 schedules[name]={'lines':sum(1 for _ in (out/name).open()),'sha256':hashlib.sha256((out/name).read_bytes()).hexdigest()}
missing=[p for p in sorted(refs) if not Path(p).is_file()]
report={'stage':'ready' if count==70283 and not missing else 'incomplete','records':count,'unique_referenced_files':len(refs),'missing_count':len(missing),'missing_examples':missing[:20],'path_map':maps,'schedules':schedules,'completed':time.time()}
(out/'data-readiness.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
assert report['stage']=='ready'
