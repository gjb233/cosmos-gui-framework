"""CPU native-template audit of real source/action representatives; no GPU."""
import pathlib,json,collections,hashlib,torch
from transformers import AutoProcessor
from PIL import Image
from mixed_prompt import render_policy,target
P=pathlib.Path(__file__).resolve().parent
actions={r['id']:r for r in map(json.loads,open(P/'artifacts/action-pair-audit.jsonl')) if r.get('parsed')}
images={r['id']:r['image'] for r in map(json.loads,open(P/'artifacts/image-decode-audit.jsonl'))}
processor=AutoProcessor.from_pretrained('/rscratch/wzzheng/code/cosmos-gui/.local/models/GUI-Libra-8B',local_files_only=True)
seen=set();results=[]
for r in map(json.loads,open(P/'artifacts/inventory.jsonl')):
 if r['id'] not in actions:continue
 a=actions[r['id']]['action'];key=(r['source'],a['action_type'])
 if key in seen:continue
 seen.add(key)
 # Template-only probe: description as text does NOT grant instruction approval.
 for level in ('low','high'):
  policy=dict(current_image=images[r['id']],instruction=a['action_description'],history=[],level=level)
  msgs=render_policy(policy)
  image=Image.open(images[r['id']]).convert('RGB')
  prefix=processor.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
  full=processor.apply_chat_template(msgs+[{'role':'assistant','content':target(a)}],tokenize=False,add_generation_prompt=False)
  x=processor(text=[prefix],images=[image],return_tensors='pt')
  y=processor(text=[full],images=[image],return_tensors='pt')
  n=x['input_ids'].shape[1]
  assert torch.equal(x['input_ids'],y['input_ids'][:,:n])
  labels=y['input_ids'].clone();labels[:,:n]=-100
  assert (labels!=-100).any() and (labels[:,:n]==-100).all()
  results.append(dict(id=r['id'],source=r['source'],action=a['action_type'],level=level,prefix_tokens=n,supervised_tokens=int((labels!=-100).sum())))
  print(len(results),key,level,flush=True)
report=dict(passed=True,cases=len(results),source_action_pairs=len(seen),results=results,prompt_sha256=hashlib.sha256((P/'mixed_prompt.py').read_bytes()).hexdigest(),
 scope='Native image/text prefix and assistant-only label boundary; descriptions used as structural probes in both levels, not approved High instructions.',training_ready=False)
(P/'artifacts/native-mixed-template-audit.json').write_text(json.dumps(report,indent=2))
