from pathlib import Path
import shutil,json,ast,time,hashlib
W=Path('/data/mai/training/mai-ui-zero2-mlp-strict10k-20261113');R=W/'run'
plan=json.loads((W/'training-plan.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(Path(plan['manifest']))==plan['manifest_sha256']
assert [sha(W/f'phase{i}.jsonl') for i in (1,2)]==plan['phase_sha256']
state=json.loads((R/'checkpoints/latest-complete.json').read_text());assert state['completed_steps']==200
source=Path('/data/mai/zero2_fulltrain_resume_hook.py');ast.parse(source.read_text())
backup=W/'zero2_hook.before-resume200.py';assert not backup.exists();shutil.copy2(W/'zero2_hook.py',backup);shutil.copy2(source,W/'zero2_hook.py')
for name in ('pause-request-200.json','exit.json'):
 p=R/name
 if p.exists():p.rename(R/(name+'.before-resume200'))
(R/'user-resume200.json').write_text(json.dumps(dict(from_step=200,next_draw=51200,requested_at=time.time(),manifest_and_schedules_verified=True)))
print('RESUME_READY',flush=True)
