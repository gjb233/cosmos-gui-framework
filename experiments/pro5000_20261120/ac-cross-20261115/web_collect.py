from pathlib import Path
import json,os
R=Path('/data/mai/evaluation/ac-cross-20261115/mind2web')/os.environ['PROMPT_ARM']
p=R/'step-000000/mind2web-predictions-0.jsonl'
a=[json.loads(l) for l in p.open()];assert len(a)==len({x['id'] for x in a})==3349
result={}
for split in ['overall','task','website','domain']:
 v=[x for x in a if split=='overall' or x['split']==split];n=len(v)
 result[split]=dict(total=n,correct=sum(x['correct'] for x in v),step_success_rate=sum(x['correct'] for x in v)/n,element_accuracy=sum(x['element_acc'] for x in v)/n,operation_f1=sum(x['operation_f1'] for x in v)/n,parse_errors=sum(x['error'] is not None for x in v),truncated=sum(t['truncated'] for x in v for t in x['traces']))
(R/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
