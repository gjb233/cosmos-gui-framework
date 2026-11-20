from pathlib import Path
import json,collections
R=Path('/data/mai/evaluation/aw-base-matched-20261118');F=Path('/data/mai/evaluation/aw-fm2-step100-20261118')
def read(root,step):
 rows=[];interactions=[]
 for slot in range(8):
  p=root/'full'/f'step-{step:06d}'/f'slot-{slot}'
  assert json.loads((p/'complete.json').read_text())['completed']
  rows.extend(json.loads((p/'results.json').read_text()))
  interactions.extend(json.loads(x) for x in (p/'interactions.jsonl').read_text().splitlines())
 assert len(rows)==116 and len({x['task_template'] for x in rows})==116
 return {x['task_template']:x for x in rows},dict(actions=len(interactions),action_errors=sum(bool(x['action_error']) for x in interactions),truncated=sum(x['finish_reason']=='length' for x in interactions),thinking=sum(x['thinking_present'] for x in interactions))
a,astats=read(R,0);b,bstats=read(F,100);assert set(a)==set(b)
mismatched=[k for k in a if a[k]['goal']!=b[k]['goal']]
improved=[k for k in a if a[k]['is_successful']!=1 and b[k]['is_successful']==1]
regressed=[k for k in a if a[k]['is_successful']==1 and b[k]['is_successful']!=1]
base=sum(x['is_successful']==1 for x in a.values());fm=sum(x['is_successful']==1 for x in b.values())
result=dict(total=116,identical_goals=not mismatched,goal_mismatch_tasks=mismatched,base_successful=base,fm2_successful=fm,base_percent=base/116*100,fm2_percent=fm/116*100,difference_percentage_points=(fm-base)/116*100,improved_tasks=improved,regressed_tasks=regressed,both_successful=[k for k in a if a[k]['is_successful']==b[k]['is_successful']==1],base_generation=astats,fm2_generation=bstats,base_environment_errors=[k for k in a if a[k]['exception_info']],fm2_environment_errors=[k for k in b if b[k]['exception_info']])
common=[k for k in a if k not in mismatched and not a[k]['exception_info'] and not b[k]['exception_info']]
result['common_valid']=dict(total=len(common),base_successful=sum(a[k]['is_successful']==1 for k in common),fm2_successful=sum(b[k]['is_successful']==1 for k in common),improved_tasks=[k for k in common if a[k]['is_successful']!=1 and b[k]['is_successful']==1],regressed_tasks=[k for k in common if a[k]['is_successful']==1 and b[k]['is_successful']!=1])
(R/'comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
