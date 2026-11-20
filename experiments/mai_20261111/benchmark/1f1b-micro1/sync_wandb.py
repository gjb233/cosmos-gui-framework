from pathlib import Path
import os,time,json,uuid,fcntl
O=Path(__file__).resolve().parent;W=O/'wandb-sync';W.mkdir(exist_ok=True)
lock=(W/'sync.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
while not (O/'active.json').exists():time.sleep(5)
import wandb
os.environ.update(WANDB_CONSOLE='off',WANDB_DISABLE_CODE='true',WANDB_SILENT='true',WANDB_WATCH='false')
state_file=W/'state.json'
state=json.loads(state_file.read_text()) if state_file.exists() else dict(id=uuid.uuid4().hex[:8],last_step=350)
run=wandb.init(entity='guojh23-tsinghua-university',project='cosmos-gui',id=state['id'],resume='allow',name='70283-resume350-1f1b7-micro4-gb56-a22',mode='online',dir=str(W),settings=wandb.Settings(x_disable_stats=True,x_disable_meta=True,disable_git=True,console='off',init_timeout=60),config=dict(framework='PyTorch Schedule1F1B',pipeline_stages=7,microbatch=4,global_batch=56,accumulation_waves=2,recompute='per layer',resume_step=350,next_draw=20800,source_wandb='2a95131b',gpus=list(range(7)),layers=[3,6,6,6,5,5,5]))
state['url']=run.url;state_file.write_text(json.dumps(state,indent=2))
run.define_metric('optimizer_step');run.define_metric('*',step_metric='optimizer_step')
def rows(f):
 if not f.exists():return {}
 result={}
 for l in f.read_text().splitlines():
  try:r=json.loads(l)
  except json.JSONDecodeError:continue
  result[r['step']]=r
 return result
while True:
 losses=rows(O/'loss.jsonl');times=[rows(O/f'timing-rank{r}.jsonl') for r in range(7)]
 for step in sorted(losses):
  if step<=state['last_step'] or any(step not in x for x in times):continue
  ts=[x[step] for x in times]
  rec=dict(optimizer_step=step,loss=losses[step]['loss'],ce=losses[step]['ce'],fm=losses[step]['fm'],seconds=max(x['seconds'] for x in ts),samples_per_second=56/max(x['seconds'] for x in ts),prepare_seconds=max(x['prepare_seconds'] for x in ts),schedule_seconds=max(x['schedule_seconds'] for x in ts),lr=ts[0]['lr'],grad_norm=ts[0]['grad_norm'],finite_gradient_tensors=sum(x['gradient_tensors'] for x in ts))
  rec.update({f'gpu{r}_peak_gib':ts[r]['peak_GiB'] for r in range(7)})
  run.log(rec,step=step);state['last_step']=step;state_file.write_text(json.dumps(state,indent=2))
 if (O/'exit.json').exists():break
 time.sleep(5)
run.finish()
