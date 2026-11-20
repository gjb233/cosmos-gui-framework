from pathlib import Path
import os,json,subprocess,time
p=Path(__file__).resolve().parent
a=json.loads((p/'active.json').read_text())
assert not Path('/proc/'+str(a['pid'])).exists(),'existing probe alive'
s=json.loads((p/'launch-spec.json').read_text())
out=p/'test-run'
log=out/'train.log'
if log.exists():log.rename(out/('train-'+str(int(time.time()))+'.log'))
env={k:v for k,v in os.environ.items() if not k.startswith(('GUI_','WANDB_'))};env.update(s['env'])
with log.open('w') as f:q=subprocess.Popen(s['command'],env=env,cwd=s['cwd'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
a.update(pid=q.pid,time=time.time());(p/'active.json').write_text(json.dumps(a,indent=2))
print(q.pid,flush=True)
