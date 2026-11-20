
from pathlib import Path
import os,json,subprocess,signal,time,fcntl
B=Path('/data/mai/eval-fm2-step100-20261117');R=Path('/data/mai/evaluation/fm2-step100-20261117');W=Path('/data/mai/training/gui-libra-awhigh-aclow-fm2-20261117')
lock=(R/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def state(phase,**kw):(R/'status.json').write_text(json.dumps(dict(phase=phase,time=time.time(),**kw)))
try:
 latest=json.loads((W/'run/checkpoints/latest-complete.json').read_text());assert latest['completed_steps']==100
 assert len(list((W/'run/checkpoints/step-000100').glob('*optim_states.pt')))==8
 active=json.loads((W/'run/active.json').read_text());pid=active['pid'];p=Path('/proc')/str(pid)/'cmdline'
 assert str(W/'train_entry.py').encode() in p.read_bytes()
 current=json.loads((W/'run/audit/timing-rank0.jsonl').read_text().splitlines()[-1])['step']
 (R/'pause.json').write_text(json.dumps(dict(pid=pid,last_completed_step=current,resume_step=100,reason='user authorized step100 evaluation and replay of unsaved steps')))
 state('pausing',last_completed_step=current)
 os.killpg(pid,signal.SIGTERM)
 deadline=time.time()+180
 while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
  assert time.time()<deadline,'GPU processes did not exit'
  time.sleep(3)
 state('merging')
 subprocess.run(['/data/mai/envs/mai/bin/python','-u',str(B/'merge.py')],check=True)
 state('evaluating')
 subprocess.run(['/data/mai/envs/mai/bin/python','-u',str(B/'web_run.py')],env=dict(os.environ,PROMPT_ARM='direct'),check=True)
 assert (R/'direct/offline-complete.json').exists()
 state('resuming')
 old=W/'run/exit.json'
 if old.exists():old.rename(W/'run/exit-before-eval100.json')
 p=subprocess.Popen(['/data/mai/envs/mai/bin/python','-u',str(W/'launch.py')],stdin=subprocess.DEVNULL,stdout=(W/'run/resume-after-eval100.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
 (R/'resume-launch.json').write_text(json.dumps(dict(pid=p.pid,step=100)))
 state('evaluation_completed_resume_launched',resume_launcher_pid=p.pid)
except Exception as e:
 state('failed',error=repr(e));raise
