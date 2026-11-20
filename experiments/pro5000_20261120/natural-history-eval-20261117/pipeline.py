
from pathlib import Path
import subprocess,json,os,signal,time,fcntl
B=Path('/data/mai/natural-history-eval-20261117');R=Path('/data/mai/evaluation/natural-history-20261117')
W=Path('/data/mai/training/gui-libra-natural-history-20261117');O=Path('/data/mai/training/gui-libra-awhigh-aclow-20261116')
lock=(R/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def state(phase,**kw):(R/'status.json').write_text(json.dumps(dict(phase=phase,time=time.time(),**kw)))
try:
 state('validating')
 subprocess.run(['/data/mai/envs/mai/bin/python','-u',str(W/'validate_processor.py')],check=True)
 assert json.loads((W/'processor-validation.json').read_text())['passed']
 state('stopping_previous_run')
 latest=json.loads((O/'run/checkpoints/latest-complete.json').read_text());pid=json.loads((O/'run/active.json').read_text())['pid']
 p=Path('/proc')/str(pid)/'cmdline';assert str(O/'train_entry.py').encode() in p.read_bytes()
 assert len(list((O/'run/checkpoints'/latest['tag']).glob('*optim_states.pt')))==8
 (R/'previous-run.json').write_text(json.dumps(dict(path=str(O),preserved_checkpoint=latest['completed_steps'],last_completed_step=json.loads((O/'run/audit/timing-rank0.jsonl').read_text().splitlines()[-1])['step'])))
 os.killpg(pid,signal.SIGTERM)
 deadline=time.time()+180
 while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
  assert time.time()<deadline;time.sleep(3)
 state('baseline_evaluation')
 subprocess.run(['/data/mai/envs/mai/bin/python','-u',str(B/'web_run.py')],env=dict(os.environ,PROMPT_ARM='direct'),check=True)
 assert (R/'direct/offline-complete.json').exists()
 p=W/'training-plan.json';d=json.loads(p.read_text());d.update(ready_to_launch=True,pending_checks=[]);p.write_text(json.dumps(d,indent=2))
 state('starting_natural_history_training')
 p=subprocess.Popen(['/data/mai/envs/mai/bin/python','-u',str(W/'launch.py')],stdin=subprocess.DEVNULL,stdout=(W/'run/launcher.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
 state('training_launched',launcher_pid=p.pid)
except Exception as e:state('failed',error=repr(e));raise
