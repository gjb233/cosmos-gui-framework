from pathlib import Path
import json,os
R=Path('/data/mai/evaluation/ac-cross-20261115/androidworld')/('round-'+os.environ['LIBRA_REPRO_ROUND'])
rows=[]
for i in range(8):
 p=R/'full/step-000000'/f'slot-{i}'
 assert json.loads((p/'complete.json').read_text())['completed']
 rows+=json.loads((p/'results.json').read_text())
assert len(rows)==len({r['task'] for r in rows})==115
summary=dict(total=115,official_code_successes=sum(r['success'] for r in rows),paper_terminate_successes=sum(r['paper_terminate_success'] for r in rows),environment_error_tasks=[r['task'] for r in rows if r['environment_error']],seed=42,max_steps=20)
(R/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
