from pathlib import Path
import json,time,os,statistics,math
W=Path('/data/mai/training/gui-libra-awhigh-aclow-20261116');R=W/'run';A=R/'audit'
def read(p):return json.loads(p.read_text()) if p.exists() else None
def rows(p):
 out=[]
 if p.exists():
  for line in p.read_text().splitlines():
   try:out.append(json.loads(line))
   except json.JSONDecodeError:pass
 return out
active=read(R/'active.json');pid=active['pid'];running=Path(f'/proc/{pid}').exists()
ranks=[rows(A/f'timing-rank{i}.jsonl') for i in range(8)]
latest=[x[-1]['step'] if x else 0 for x in ranks];common=min(latest)
current=[]
if common:
 current=[next(x for x in reversed(rank) if x['step']==common) for rank in ranks]
recent=ranks[0][-10:];seconds=statistics.mean(x['seconds'] for x in recent) if recent else None
checkpoint=read(R/'checkpoints/latest-complete.json');saved=checkpoint['completed_steps'] if checkpoint else 0
last_activity=max((p.stat().st_mtime for p in A.glob('draws-rank*.jsonl')),default=0)
issues=[]
pause=read(R/f'pause-request-{saved}.json')
paused=bool(pause and not running and common==saved and pause.get('checkpoint')==f'step-{saved:06d}')
if not running and common<725 and not paused:issues.append('training_process_exited_before_completion')
if running and time.time()-last_activity>900:issues.append('no_new_microbatch_for_15_minutes')
if any(not math.isfinite(x['loss']) for rank in ranks for x in rank):issues.append('nonfinite_loss')
expected=(common//50)*50 if common>=50 else 2 if common>=2 else 0
if saved<expected and common>expected:issues.append('checkpoint_missing_after_save_step')
out=dict(checked_at=time.strftime('%Y-%m-%d %H:%M:%S %Z'),running=running,paused_by_user=paused,pid=pid,completed_steps=common,total_steps=725,
 per_rank_steps=latest,latest_checkpoint=saved,exit=read(R/'exit.json'),issues=issues,
 global_mean_loss=statistics.mean(x['loss'] for x in current) if current else None,
 initial_global_mean_loss=statistics.mean(rank[0]['loss'] for rank in ranks) if common else None,
 recent_seconds_per_step=seconds,estimated_remaining_hours=(725-common)*seconds/3600 if seconds else None,
 peak_allocated_gib=max((x['peak_allocated_gib'] for rank in ranks for x in rank),default=0),
 seconds_since_data_progress=time.time()-last_activity)
if common==725 and not running:
 completes=[read(A/f'complete-rank{i}.json') for i in range(8)]
 out['completion_verified']=all(c and c['passed'] and c['optimizer_steps']==725 for c in completes) and saved==725 and out['exit'] and out['exit']['exit_code']==0
print(json.dumps(out))
