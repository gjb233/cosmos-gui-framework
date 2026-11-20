
from pathlib import Path
import os,json,subprocess,signal,time,fcntl
B=Path('/data/mai/history-params-eval-20261117');R=Path('/data/mai/evaluation/history-params-20261117');W=Path('/data/mai/training/gui-libra-awhigh-aclow-20261116')
lock=(R/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def state(phase,**kw):(R/'status.json').write_text(json.dumps(dict(phase=phase,time=time.time(),**kw)))
try:
 latest=json.loads((W/'run/checkpoints/latest-complete.json').read_text());resume_step=latest['completed_steps']
 assert len(list((W/'run/checkpoints'/latest['tag']).glob('*optim_states.pt')))==8
 active=json.loads((W/'run/active.json').read_text());pid=active['pid'];p=Path('/proc')/str(pid)/'cmdline'
 assert str(W/'train_entry.py').encode() in p.read_bytes()
 current=json.loads((W/'run/audit/timing-rank0.jsonl').read_text().splitlines()[-1])['step']
 (R/'pause.json').write_text(json.dumps(dict(pid=pid,last_completed_step=current,resume_step=resume_step,reason='user authorized step300 evaluation and replay of unsaved steps')))
 state('pausing',last_completed_step=current)
 os.killpg(pid,signal.SIGTERM)
 deadline=time.time()+180
 while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
  assert time.time()<deadline,'GPU processes did not exit'
  time.sleep(3)
 for arm,step in [('base',0),('step300',300)]:
  state('evaluating',arm=arm,resume_step=resume_step)
  subprocess.run(['/data/mai/envs/mai/bin/python','-u',str(B/'web_run.py')],env=dict(os.environ,PROMPT_ARM=arm,EVAL_STEP=str(step)),check=True)
  assert (R/arm/'offline-complete.json').exists()
 state('resuming')
 old=W/'run/exit.json'
 if old.exists():old.rename(W/'run/exit-before-history-eval.json')
 p=subprocess.Popen(['/data/mai/envs/mai/bin/python','-u',str(W/'launch.py')],stdin=subprocess.DEVNULL,stdout=(W/'run/resume-after-history-eval.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
 (R/'resume-launch.json').write_text(json.dumps(dict(pid=p.pid,step=resume_step)))
 state('evaluation_completed_resume_launched',resume_launcher_pid=p.pid)
except Exception as e:
 state('failed',error=repr(e));raise
