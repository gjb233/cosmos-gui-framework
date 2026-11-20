import pathlib,json,collections,re,hashlib
P=pathlib.Path(__file__).resolve().parent
testpath=P/'originals/web-multi_test_data.json'
test=json.loads(testpath.read_text());idx=collections.defaultdict(list)
for n in test:idx[n['image_id']].append(n)
images={r['id']:r for r in map(json.loads,(P/'artifacts/image-decode-audit.jsonl').open())}
counts=collections.Counter();out=[]
for r in map(json.loads,(P/'artifacts/inventory.jsonl').open()):
 if not r['source'].startswith('guiact'):continue
 uid=pathlib.Path(r['image']).stem
 if uid not in idx:continue
 human='\n'.join(x['value'] for x in r['prefix'] if x['from']=='human')
 m=re.search(r'Instruction:\s*(.*?)(?:\n\n(?:Interaction History|Previous actions):|$)',human,re.S)
 goal=m[1].strip() if m else None
 from PIL import Image
 with Image.open(images[r['id']]['image']) as im:size=im.size
 hits=[n for n in idx[uid] if n['question'].strip()==goal and (n['image_size']['width'],n['image_size']['height'])==size]
 status='confirmed_official_test_uid_goal_size' if len(hits)==1 else 'requires_review'
 counts[status]+=1
 out.append(dict(id=r['id'],source=r['source'],image_uid=uid,status=status,excluded_from_training=len(hits)==1,goal=goal,image_size=size,original_annotation=str(testpath),original_annotation_sha256=hashlib.sha256(testpath.read_bytes()).hexdigest(),matched_records=len(hits)))
with (P/'artifacts/guiact-official-test-confirmed-exclusions.jsonl').open('w') as f:
 for r in out:f.write(json.dumps(r,ensure_ascii=False)+'\n')
report=dict(counts=counts,rows=len(out),note='Official test annotation UID, exact stripped task text and screenshot dimensions matched. Never grants train approval.')
(P/'artifacts/guiact-official-test-confirmed-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
