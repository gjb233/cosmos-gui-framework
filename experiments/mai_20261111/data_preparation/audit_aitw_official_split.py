import json,pathlib,re,collections,hashlib
P=pathlib.Path(__file__).resolve().parent
split=P/'originals/aitw-standard-split.json'
spl=json.loads(split.read_text())
index={str(e):k for k,ids in spl.items() for e in ids}
counts=collections.Counter();out=[]
for r in map(json.loads,(P/'artifacts/inventory.jsonl').open()):
 if r['source']!='aitw':continue
 name=pathlib.Path(r['image']).name
 m=re.fullmatch(r'(general|install|googleapps|google_apps|webshopping|web_shopping|single)_(\d+)[._].*',name)
 ep=m[2] if m else None;label=index.get(ep,'unknown')
 counts[label]+=1;out.append(dict(id=r['id'],source='aitw',image=r['image'],episode_id=ep,official_standard_split=label,identifier_only=True,original_action_verified=False))
with (P/'artifacts/aitw-official-split-identifier-audit.jsonl').open('w') as f:
 for r in out:f.write(json.dumps(r)+'\n')
report=dict(rows=len(out),counts=counts,split_sha256=hashlib.sha256(split.read_bytes()).hexdigest(),scope='Official Google standard split membership by source episode identifier; original episode binding still required',training_ready=False)
(P/'artifacts/aitw-official-split-identifier-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
