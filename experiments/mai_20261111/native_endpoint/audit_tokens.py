from pathlib import Path
import sys,json,ast,collections
N=Path(__file__).resolve().parent;sys.path.insert(0,str(N))
from native_protocol import *
from mixed_native_targets import tokenize_targets
from transformers import AutoProcessor
proc=AutoProcessor.from_pretrained('/rscratch/wzzheng/experiments/gui-mai-lowres-abd-a23-20261005-v1/model',local_files_only=True)
rows=[json.loads(l) for l in (N/'artifacts/train-manifest.jsonl').open()]
count=0;max_target=0
for r in rows:
 for level,v in r['views'].items():
  t=v['native_target'];ids=tokenize_targets(proc.tokenizer,[t])[0];max_target=max(max_target,len(ids))
  count+=1
 # No target/future metadata can appear through renderer allowlist.
 for lev in r['views']:
  p=dict(goal=r['goal'],instruction=r['views'][lev]['instruction'],history=r['history_mobile_use'],level=lev,current_image=r['current_image'])
  messages=render_policy(p)
  s=proc.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
  assert s.count('<|vision_start|>')==1
  assert s.endswith('<|im_start|>assistant\n'),repr(s[-150:])
  assert ('Current step query:' in messages[1]['content'][0]['text'])==(lev=='low')
out={'passed':True,'records':len(rows),'targets':count,'max_target_tokens':max_target,'no_thinking_assistant_prefix':True,'image_count':1}
(N/'token-audit.json').write_text(json.dumps(out,indent=2));print(out)
