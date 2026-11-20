
from pathlib import Path
import json,subprocess,time,os,hashlib,shutil
W=Path('/data/mai/training/gui-libra-natural-history-full-20261117')
py='/data/mai/envs/mai/bin/python'
def state(phase,**kw):(W/'preflight-status.json').write_text(json.dumps(dict(phase=phase,time=time.time(),**kw)))
try:
 state('waiting_length_audit')
 deadline=time.time()+1800
 while not (W/'token-admission.json').exists():
  assert time.time()<deadline
  time.sleep(5)
 state('processor_validation')
 subprocess.run([py,'-u',str(W/'validate_processor.py')],check=True)
 S=W/'longest-smoke';S.mkdir(exist_ok=False);(S/'run/audit').mkdir(parents=True)
 for p in W.iterdir():
  if p.is_file() and (p.suffix in ('.py','.toml') or p.name in ('config.json','training-plan.json','templates.json','processor-validation.json')):
   (S/p.name).write_text(p.read_text().replace(str(W),str(S)))
 (S/'train-manifest.jsonl').symlink_to(W/'train-manifest.jsonl')
 lengths=[json.loads(l) for l in (W/'input-lengths.jsonl').open()]
 records={r['id']:r for r in map(json.loads,(W/'train-manifest.jsonl').open())}
 longest=max(lengths,key=lambda x:x['prefix_tokens'])
 fm=max((x for x in lengths if records[x['id']]['fm_mask']),key=lambda x:x['prefix_tokens'])
 for i,ex in enumerate([longest,fm],1):
  r=records[ex['id']]
  draws=[dict(draw=j,id=r['id'],source=r['source'],level=ex['level'],variant='direct',fm_mask=r['fm_mask'],padding=False) for j in range(256)]
  (S/f'phase{i}.jsonl').write_text(''.join(json.dumps(d)+'\n' for d in draws))
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 plan=json.loads((S/'training-plan.json').read_text());plan.update(optimizer_steps=2,warmup_steps=1,original_draws=512,used_draws=512,dropped_final_incomplete_batch=0,ready_to_launch=True,pending_checks=[],phase_sha256=[sha(S/f'phase{i}.jsonl') for i in (1,2)])
 (S/'training-plan.json').write_text(json.dumps(plan,indent=2))
 cfg=json.loads((S/'config.json').read_text());cfg['train']['max_iter']=2
 for i in (1,2):cfg['sampling'][f'phase{i}']['draws']=256
 (S/'config.json').write_text(json.dumps(cfg,indent=2))
 p=S/'native.toml';s=p.read_text()
 import re
 s=re.sub(r'max_iter = \d+','max_iter = 2',s);s=re.sub(r'cycle_lengths = \[\d+\]','cycle_lengths = [2]',s);s=re.sub(r'warm_up_steps = \[\d+\]','warm_up_steps = [1]',s);p.write_text(s)
 # Two nonzero updates validate backward/optimizer paths; smoke state is never reused.
 p=S/'zero2_hook.py';s=p.read_text();a="lr=1e-5*(step/warmup if step<warmup else (total-1-step)/max(1,total-1-warmup))";assert a in s;s=s.replace(a,'lr=1e-5');p.write_text(s)
 state('longest_sample_smoke',longest=longest,longest_fm=fm)
 subprocess.run([py,'-u',str(S/'launch.py')],check=True)
 assert json.loads((S/'run/exit.json').read_text())['exit_code']==0
 assert json.loads((S/'run/checkpoints/latest-complete.json').read_text())['completed_steps']==2
 assert all((S/f'run/audit/complete-rank{i}.json').exists() for i in range(8))
 (W/'longest-smoke-validation.json').write_text(json.dumps(dict(passed=True,longest=longest,longest_fm=fm,world_size=8,global_batch=256,microbatch=8,optimizer_steps=2)))
 p=W/'training-plan.json';plan=json.loads(p.read_text());plan.update(ready_to_launch=True,pending_checks=[]);p.write_text(json.dumps(plan,indent=2))
 state('starting_training')
 proc=subprocess.Popen([py,'-u',str(W/'launch.py')],stdin=subprocess.DEVNULL,stdout=(W/'run/launcher.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
 state('training_launched',launcher_pid=proc.pid)
except Exception as e:
 state('failed',error=repr(e));raise
