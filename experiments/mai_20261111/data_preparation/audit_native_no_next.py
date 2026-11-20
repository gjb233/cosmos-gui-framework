import pathlib,json,hashlib
from mixed_native_sample import package
p=pathlib.Path(__file__).resolve().parent
actions={r['id']:r for r in map(json.loads,open(p/'artifacts/action-pair-audit.jsonl')) if r.get('parsed')}
images={r['id']:r['image'] for r in map(json.loads,open(p/'artifacts/image-decode-audit.jsonl'))}
seen=set();cases=[]
for r in map(json.loads,open(p/'artifacts/inventory.jsonl')):
 if r['id'] not in actions:continue
 a=actions[r['id']]['action'];key=(r['source'],a['action_type'])
 if key in seen:continue
 seen.add(key)
 record=dict(id=r['id'],current_image=images[r['id']],next_image=None,fm_mask=0,
 terminal=a['action_type']=='Terminate',history=[],views={'low':dict(instruction=a['action_description'],action=a)})
 try:package(record,'low')
 except ValueError as e:assert str(e)=='unapproved training record'
 else:raise AssertionError('approval gate failed')
 sample=package(record,'low',probe=True)
 assert json.loads(sample['gui_action_text'])['action_type']==a['action_type']
 assert sample['gui_fm_mask']==0
 cases.append(dict(source=r['source'],action=a['action_type'],terminal=record['terminal'],video_shape=list(sample['video'].shape)))
report=dict(passed=True,cases=cases,count=len(cases),nonterminal_no_next=sum(not r['terminal'] for r in cases),approval_gate_passed=True,scope='CPU packaging only; no distributed missing-next backward verification',training_ready=False)
(p/'artifacts/native-no-next-packaging-audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='cases'}),flush=True)
