from pathlib import Path
import json,sys,importlib,os
R=Path('/data/mai/evaluation/natural-history-20261117')/os.environ['PROMPT_ARM'];A=Path('/data/mai/evaluation/assets-20261113')
summary={}
for step in (0,):
 out=R/f'step-{step:06d}'
 if not out.exists():continue
 entry={}
 for bench in ('grounding','mind2web'):
  rows=[json.loads(l) for l in (A/bench/'manifest.jsonl').open()]
  if bench=='grounding':rows=[r for r in rows if r['dataset'] in ('androidcontrol-high','androidcontrol-low')]
  expected={x['id'] for x in rows}
  predictions=[json.loads(l) for p in out.glob(f'{bench}-predictions-*.jsonl') for l in p.open()]
  ids=[p['id'] for p in predictions];assert len(ids)==len(set(ids)),(step,bench,'duplicates')
  assert set(ids)<=expected
  entry[bench]=dict(completed=len(ids),total=len(rows))
  if set(ids)!=expected:continue
  done={x['id']:x for x in predictions}
  if bench=='grounding':
   sys.path.insert(0,str(A/bench));from official_eval import process_file
   scores={}
   for dataset in ('androidcontrol-high','androidcontrol-low'):
    values=[done[r['id']] for r in rows if r['dataset']==dataset];d=out/dataset;d.mkdir(exist_ok=True)
    for field in ('plan','ans'):(d/f'{field}.jsonl').write_text(''.join(json.dumps(v[field])+'\n' for v in values))
    score=dict(process_file(str(A/bench/'samples.json'),str(d/'plan.jsonl'),str(d/'ans.jsonl')))
    assert score['total_steps']==398,score
    score.update(accuracy=score['correct_steps']/score['total_steps'],parse_errors=sum(x['native_parse_error'] for x in values),truncated=sum(x['truncated'] for x in values));scores[dataset]=score
  else:
   scores={}
   for split in ('overall','task','website','domain'):
    values=[done[r['id']] for r in rows if split=='overall' or r['split']==split];n=len(values)
    scores[split]=dict(total=n,step_success_rate=sum(x['correct'] for x in values)/n,element_accuracy=sum(x['element_acc'] for x in values)/n,operation_f1=sum(x['operation_f1'] for x in values)/n,parse_errors=sum(x['error'] is not None for x in values),truncated_requests=sum(t['truncated'] for x in values for t in x['traces']))
  entry[bench]['results']=scores
 summary[str(step)]=entry
(R/'comparison.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

