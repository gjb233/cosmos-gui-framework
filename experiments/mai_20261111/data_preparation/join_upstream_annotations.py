
import json,hashlib,collections
from pathlib import Path
p=Path(__file__).resolve().parent
root=Path('/rscratch/wzzheng/data/mai/aguvis-stage2')
mapping={'aitw':'aitw-l1.json','amex':'amex-l1.json','android_control':'android_control.json','coat-terminal':'coat.json','guiact-web':'guiact-web-multi-l1.json','guiact-web-chinese':'guiact-web-multi-l1.json','mind2web':'mind2web-l1.json','gui-odyssey':'gui-odyssey-l1.json'}
inv=collections.defaultdict(list)
for line in (p/'artifacts/inventory.jsonl').open():
 r=json.loads(line);inv[r['source']].append((r['id'],r['image']))
report={}
with (p/'artifacts/upstream-annotation-joins.jsonl').open('w') as out:
 for source,name in mapping.items():
  raw=(root/name).read_bytes();sha=hashlib.sha256(raw).hexdigest();rows=json.loads(raw);del raw
  index=collections.defaultdict(list)
  for i,row in enumerate(rows):
   for im in (row['image'] if isinstance(row['image'],list) else [row['image']]):index[im].append(i)
  counts=collections.Counter()
  for uid,image in inv[source]:
   hits=index.get(image,[]);counts['matched' if hits else 'missing']+=1
   if len(hits)>1:counts['ambiguous']+=1
   matches=[]
   for i in hits:
    row=rows[i]
    matches.append(dict(row_index=i,row_sha256=hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),conversations=row['conversations']))
   out.write(json.dumps(dict(id=uid,source=source,image=image,annotation_path=str(root/name),annotation_sha256=sha,matches=matches,original_episode_verified=False,transition_verified=False),ensure_ascii=False)+chr(10))
  out.flush();report[source]=dict(counts)
  (p/'artifacts/upstream-join-report.json').write_text(json.dumps(dict(sources=report,training_ready=False,note='Exact upstream image-key joins; not original trajectory alignment or heldout proof'),indent=2))
  print(source,dict(counts),flush=True)
  del rows,index
