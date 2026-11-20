from pathlib import Path
import sys,json,math,hashlib
W=Path('/data/mai/training/gui-libra-natural-history-full-20261117/longest-smoke');sys.path.insert(0,str(W))
from native_protocol import render_policy,target,parse,HIGH_FIELDS,LOW_FIELDS
from training_backend import validate_inputs
from mixed_native_targets import tokenize_targets
from transformers import AutoProcessor
from PIL import Image
import torch
torch.set_num_threads(4)
cfg,records,draws=validate_inputs(W/'config.json',W/'train-manifest.jsonl',[W/'phase1.jsonl',W/'phase2.jsonl'])
p=AutoProcessor.from_pretrained('/data/mai/benchmarks/gui-libra-base/ready-model',local_files_only=True)
examples=json.loads((W/'prompt-examples.json').read_text());audits=[]
for ex in examples:
 r=records[ex['id']];level=ex['level'];v=r['views'][level]
 with Image.open(r['current_image']) as opened:im=opened.convert('RGB')
 policy=dict(current_image=im,instruction=v['instruction'],history=r['history'],level=level,goal=r['goal'])
 messages=render_policy(policy);messages[1]['content'].reverse()
 prompt=p.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
 inp=p(text=[prompt],images=[im],return_tensors='pt',images_kwargs={'size':{'shortest_edge':1024,'longest_edge':max(16777216,math.ceil(im.height/32)*32*math.ceil(im.width/32)*32)}})
 actual=inp['input_ids'].shape[-1];assert actual==ex['prefix_tokens'],(r['id'],actual,ex['prefix_tokens'])
 decoded=p.tokenizer.decode(inp['input_ids'][0]);assert messages[1]['content'][1]['text'] in decoded
 assert decoded.count('<|vision_start|>')==1 and decoded.rstrip().endswith('<|im_start|>assistant')
 answer=target(v['action'],level);fields=parse(answer)['fields']
 assert set(fields)==set(HIGH_FIELDS if level=='high' else LOW_FIELDS)
 ids,weights=tokenize_targets(p.tokenizer,[answer],True)
 assert len(ids[0])==ex['target_tokens'] and abs(sum(weights[0])-1)<1e-9
 # High prompt cannot depend on the current action, low instruction or reasoning label.
 if level=='high':
  altered=dict(policy,instruction='MUST_NOT_APPEAR_CURRENT_ANSWER')
  assert render_policy(altered)[1]['content'][0]['text']==messages[1]['content'][1]['text']
 audits.append(dict(id=r['id'],source=r['source'],level=level,prefix_tokens=actual,target_tokens=len(ids[0]),answer_fields=list(fields)))
 print('PROCESSOR_VERIFIED',r['source'],level,actual,len(ids[0]),flush=True)
report=dict(passed=True,records=len(records),draws=len(draws),checks=audits,high_current_label_independence=True,prefix_not_supervised=True,target_roundtrip=True)
(W/'processor-validation.json').write_text(json.dumps(report,indent=2))
print('CPU_VALIDATION_PASSED',len(audits),flush=True)

