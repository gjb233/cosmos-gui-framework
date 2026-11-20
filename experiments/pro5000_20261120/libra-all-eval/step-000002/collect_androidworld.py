from pathlib import Path
import json,math
R=Path('/data/mai/evaluation/gui-libra-all-20261114/step-000002/androidworld')
rows=[];selected=[];expected=None
for slot in range(8):
 p=R/'full/step-000002'/f'slot-{slot}'
 assert json.loads((p/'complete.json').read_text())['completed']
 tasks=json.loads((p/'tasks.json').read_text())
 if expected is None:expected=tasks['all_tasks']
 assert expected==tasks['all_tasks']
 selected+=tasks['selected'];part=json.loads((p/'results.json').read_text());assert isinstance(part,list)
 assert {x['task_template'] for x in part}==set(tasks['selected'])
 rows+=part
assert len(expected)==116 and len(selected)==len(set(selected))==116 and set(selected)==set(expected)
assert len(rows)==116
valid=[r for r in rows if isinstance(r.get('is_successful'),(int,float,bool)) and math.isfinite(r['is_successful'])]
success=sum(r['is_successful']==1 for r in valid)
errors=[r['task_template'] for r in rows if r.get('exception_info') or r not in valid]
summary=dict(step=2,seed=30,total=116,successful=success,success_rate_all_tasks=success/116,valid_tasks=len(valid),success_rate_valid_tasks=success/len(valid) if valid else None,environment_error_tasks=errors)
(R/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
